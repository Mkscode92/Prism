from __future__ import annotations

import ast
import hashlib
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import git
import voyageai
from pinecone import Pinecone

from config import settings

logger = logging.getLogger("prism.rag.indexer")

_voyage = voyageai.Client(api_key=settings.voyage_api_key)
_pc = Pinecone(api_key=settings.pinecone_api_key)

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt"}
VOYAGE_BATCH_SIZE = 128
MIN_FUNCTION_LINES = 3

# Regex for named JS/TS functions (captures both `function foo()` and `const foo = () =>`)
JS_FUNCTION_RE = re.compile(
    r"(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:function\s+(\w+)\s*\([^)]*\)"  # function foo(...)
    r"|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[\w]+)\s*=>)"  # foo = (...) =>
)

# Regex for Java/Kotlin methods
JAVA_METHOD_RE = re.compile(
    r"(?:(?:public|private|protected|static|final|override|suspend|abstract)\s+)+"
    r"(?:[\w<>\[\],\s?]+?\s+)"   # return type
    r"(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{"  # methodName(...) {
)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class FunctionChunk:
    repo_url: str
    file_path: str        # Relative to repo root
    function_name: str
    source_text: str
    start_line: int
    end_line: int

    @property
    def chunk_id(self) -> str:
        key = f"{self.repo_url}::{self.file_path}::{self.function_name}"
        return hashlib.sha256(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_repo_indexed(repo_url: str) -> str:
    """
    Returns the Pinecone namespace for this repo.
    Clones and indexes the repo only if the namespace doesn't exist yet.
    """
    namespace = _namespace_for(repo_url)
    if _namespace_exists(namespace):
        logger.info("Repo already indexed in namespace=%s", namespace)
        return namespace

    logger.info("Indexing repo %s into namespace=%s", repo_url, namespace)
    count = _index_repo(repo_url, namespace)
    logger.info("Indexed %d chunks for repo=%s", count, repo_url)
    return namespace


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _namespace_for(repo_url: str) -> str:
    return hashlib.sha256(repo_url.encode()).hexdigest()[:16]


def _namespace_exists(namespace: str) -> bool:
    index = _pc.Index(settings.pinecone_index)
    stats = index.describe_index_stats()
    ns = stats.namespaces or {}
    return namespace in ns and ns[namespace].vector_count > 0


def _index_repo(repo_url: str, namespace: str) -> int:
    tmp = tempfile.mkdtemp()
    try:
        git.Repo.clone_from(repo_url, tmp, depth=1)
        chunks = _collect_chunks(repo_url, Path(tmp))
        if not chunks:
            logger.warning("No function chunks found in repo %s", repo_url)
            return 0
        _embed_and_upsert(chunks, namespace)
        return len(chunks)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _collect_chunks(repo_url: str, repo_root: Path) -> list[FunctionChunk]:
    chunks: list[FunctionChunk] = []
    for path in repo_root.rglob("*"):
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        # Skip common non-source directories
        rel = path.relative_to(repo_root)
        parts = rel.parts
        if any(p in {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build"} for p in parts):
            continue

        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        rel_str = str(rel).replace("\\", "/")
        if path.suffix == ".py":
            chunks.extend(_chunk_python(repo_url, rel_str, source))
        elif path.suffix in {".java", ".kt"}:
            chunks.extend(_chunk_java(repo_url, rel_str, source))
        else:
            chunks.extend(_chunk_js(repo_url, rel_str, source))

    return chunks


def _chunk_python(repo_url: str, file_path: str, source: str) -> list[FunctionChunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.debug("Skipping unparseable file: %s", file_path)
        return []

    lines = source.splitlines()
    chunks: list[FunctionChunk] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not hasattr(node, "end_lineno"):
            continue
        length = node.end_lineno - node.lineno
        if length < MIN_FUNCTION_LINES:
            continue

        body = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        chunks.append(
            FunctionChunk(
                repo_url=repo_url,
                file_path=file_path,
                function_name=node.name,
                source_text=body,
                start_line=node.lineno,
                end_line=node.end_lineno,
            )
        )

    return chunks


def _chunk_java(repo_url: str, file_path: str, source: str) -> list[FunctionChunk]:
    lines = source.splitlines()
    chunks: list[FunctionChunk] = []
    seen_names: set[str] = set()

    for match in JAVA_METHOD_RE.finditer(source):
        name = match.group(1)
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        start_line = source[: match.start()].count("\n") + 1
        # Walk forward to find the closing brace
        depth = 0
        end_line = start_line
        for i, line in enumerate(lines[start_line - 1:], start=start_line):
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                end_line = i
                break
        else:
            end_line = min(start_line + 60, len(lines))

        if end_line - start_line < MIN_FUNCTION_LINES:
            continue

        body = "\n".join(lines[start_line - 1 : end_line])
        chunks.append(
            FunctionChunk(
                repo_url=repo_url,
                file_path=file_path,
                function_name=name,
                source_text=body,
                start_line=start_line,
                end_line=end_line,
            )
        )

    return chunks


def _chunk_js(repo_url: str, file_path: str, source: str) -> list[FunctionChunk]:
    lines = source.splitlines()
    chunks: list[FunctionChunk] = []
    seen_names: set[str] = set()

    for match in JS_FUNCTION_RE.finditer(source):
        name = match.group(1) or match.group(2)
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        # Find which line the match starts on
        start_line = source[: match.start()].count("\n") + 1
        end_line = min(start_line + 30, len(lines))

        if end_line - start_line < MIN_FUNCTION_LINES:
            continue

        body = "\n".join(lines[start_line - 1 : end_line])
        chunks.append(
            FunctionChunk(
                repo_url=repo_url,
                file_path=file_path,
                function_name=name,
                source_text=body,
                start_line=start_line,
                end_line=end_line,
            )
        )

    return chunks


def _embed_and_upsert(chunks: list[FunctionChunk], namespace: str) -> None:
    index = _pc.Index(settings.pinecone_index)

    for batch_start in range(0, len(chunks), VOYAGE_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + VOYAGE_BATCH_SIZE]
        texts = [c.source_text for c in batch]

        result = _voyage.embed(texts, model=settings.voyage_model, input_type="document")
        embeddings = result.embeddings

        vectors = [
            {
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": {
                    "repo_url": chunk.repo_url,
                    "file_path": chunk.file_path,
                    "function_name": chunk.function_name,
                    "source_text": chunk.source_text[:2000],  # Pinecone metadata limit
                    "start_line": chunk.start_line,
                },
            }
            for chunk, embedding in zip(batch, embeddings)
        ]

        index.upsert(vectors=vectors, namespace=namespace)
        logger.debug("Upserted %d vectors (batch %d)", len(vectors), batch_start // VOYAGE_BATCH_SIZE)
