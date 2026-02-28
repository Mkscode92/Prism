from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Literal
from google import genai
from google.genai import types 
from config import settings

logger = logging.getLogger("prism.classifier")
client = genai.Client(api_key=settings.gemini_api_key)

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



CLASSIFY_FUNC = types.FunctionDeclaration(
    name="classify_review",
    description="Classify a Google Play Store review into one intent category.",
    parameters={
        "type": "OBJECT", 
        "properties": {
            "intent": {
                "type": "STRING",
                "enum": ["bug", "feature", "ux", "vague"],
                "description": "Primary intent: 'bug' for crashes, 'feature' for requests, etc."
            },
            "is_vague": {"type": "BOOLEAN"},
            "confidence": {"type": "NUMBER"},
            "reasoning": {"type": "STRING"},
        },
        "required": ["intent", "is_vague", "confidence", "reasoning"],
    },
)

FOLLOW_UP_FUNC = types.FunctionDeclaration(
    name="generate_follow_up_questions",
    description="Generate 3 targeted follow-up questions for a vague review.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "questions": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            }
        },
        "required": ["questions"],
    },
)

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
    
    #this is where the review gets classified into one of 4 categories via Gemini 
    config = types.GenerateContentConfig(
        system_instruction=CLASSIFY_SYSTEM,
        tools=[types.Tool(function_declarations=[CLASSIFY_FUNC])],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY", 
                allowed_function_names=["classify_review"]
            )
        )
    )

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=f"Star rating: {star_rating}/5\n\nReview:\n{review_text}",
        config=config
    )

    fc = response.candidates[0].content.parts[0].function_call
    data = fc.args
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
    config = types.GenerateContentConfig(
        system_instruction=FOLLOW_UP_SYSTEM,
        tools=[types.Tool(function_declarations=[FOLLOW_UP_FUNC])],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY", 
                allowed_function_names=["generate_follow_up_questions"]
            )
        )
    )

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=f"Vague review:\n{review_text}",
        config=config
    )

    args = response.candidates[0].content.parts[0].function_call.args
    return list(args["questions"])