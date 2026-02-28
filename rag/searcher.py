from __future__ import annotations

import logging

from pinecone import Pinecone

from config import settings
from fix_generator import CodeChunk
from rag.indexer import _namespace_for, _voyage

logger = logging.getLogger("prism.rag.searcher")

_pc = Pinecone(api_key=settings.pinecone_api_key)


def search_code(review_text: str, repo_url: str) -> list[CodeChunk]:
    """
    Embed the review text and perform a semantic search against the indexed repo.
    Returns chunks ordered by descending similarity score.
    """
    namespace = _namespace_for(repo_url)

    result = _voyage.embed(
        [review_text],
        model=settings.voyage_model,
        input_type="query",
    )
    query_vector = result.embeddings[0]

    index = _pc.Index(settings.pinecone_index)
    response = index.query(
        vector=query_vector,
        top_k=settings.pinecone_top_k,
        namespace=namespace,
        include_metadata=True,
    )

    chunks = [
        CodeChunk(
            file_path=match.metadata["file_path"],
            function_name=match.metadata["function_name"],
            source_text=match.metadata["source_text"],
            score=match.score,
        )
        for match in response.matches
    ]

    logger.debug("Found %d code chunks for query in namespace=%s", len(chunks), namespace)
    return chunks
