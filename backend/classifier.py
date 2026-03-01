from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Literal
import anthropic
from .config import settings

logger = logging.getLogger("prism.classifier")
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


@dataclass
class ClassificationResult:
    review_id: str
    review_text: str
    intent: Literal["bug", "feature", "ux", "vague"]
    is_vague: bool
    confidence: float
    reasoning: str
    follow_up_questions: list[str] = field(default_factory=list)
    star_rating: int | None = None


CLASSIFY_TOOL = {
    "name": "classify_review",
    "description": "Classify a Google Play Store review into one intent category.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["bug", "feature", "ux", "vague"],
                "description": "Primary intent: 'bug' for crashes, 'feature' for requests, etc.",
            },
            "is_vague": {"type": "boolean"},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["intent", "is_vague", "confidence", "reasoning"],
    },
}

FOLLOW_UP_TOOL = {
    "name": "generate_follow_up_questions",
    "description": "Generate 3 targeted follow-up questions for a vague review.",
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["questions"],
    },
}

CLASSIFY_SYSTEM = """
You are a senior mobile QA engineer analyzing Google Play Store reviews.

Classify the review as:
- bug: describes a crash, error, broken functionality, or data loss
- feature: requests new functionality not currently present
- ux: describes friction, confusion, or poor usability (not broken, just bad UX)
- vague: too brief, off-topic, or impossible to act on without more context

Set is_vague=True if the review is so ambiguous that even knowing the intent
doesn't provide enough signal to search a codebase. Example of NOT vague:
"crashes after tapping login on Android 14". Example of vague: "worst app ever".
""".strip()

FOLLOW_UP_SYSTEM = """
You generate targeted follow-up questions for vague mobile app reviews.
Your questions must help a developer reproduce or understand the issue.
Focus on: exact steps to reproduce, device model and OS version,
what was expected vs what actually happened, and how often it occurs.
Return exactly 3 concise questions.
""".strip()


def classify_review(
    review_id: str,
    review_text: str,
    star_rating: int | None,
) -> ClassificationResult:
    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=CLASSIFY_SYSTEM,
        tools=[CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "classify_review"},
        messages=[
            {"role": "user", "content": f"Star rating: {star_rating}/5\n\nReview:\n{review_text}"}
        ],
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    data = tool_use.input

    follow_ups: list[str] = []
    if data["is_vague"]:
        follow_ups = _generate_follow_up_questions(review_text)

    return ClassificationResult(
        review_id=review_id,
        review_text=review_text,
        intent=data["intent"],
        is_vague=data["is_vague"],
        confidence=data["confidence"],
        reasoning=data["reasoning"],
        follow_up_questions=follow_ups,
        star_rating=star_rating,
    )


def _generate_follow_up_questions(review_text: str) -> list[str]:
    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=512,
        system=FOLLOW_UP_SYSTEM,
        tools=[FOLLOW_UP_TOOL],
        tool_choice={"type": "tool", "name": "generate_follow_up_questions"},
        messages=[
            {"role": "user", "content": f"Vague review:\n{review_text}"}
        ],
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    return list(tool_use.input["questions"])
