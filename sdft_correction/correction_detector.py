"""Detects when a user corrects the model in a conversation."""
import json
import re
from dataclasses import dataclass


@dataclass
class CorrectionResult:
    is_correction: bool
    what_was_wrong: str
    what_should_be: str
    original_model_response: str
    raw_llm_output: str


DETECTION_SYSTEM_PROMPT = """\
You are a conversation analyst. Determine whether the user's latest message is \
correcting something the assistant said previously.

A correction means the user is telling the assistant it made a mistake, gave wrong \
information, listed steps in the wrong order, or otherwise got something wrong, AND \
is providing the correct information.

NOT corrections: follow-up questions, requests for more detail, new topics, simple \
acknowledgements, opinion disagreements.

Respond with EXACTLY this JSON (no other text):
{"is_correction": true/false, "what_was_wrong": "description of the error", \
"what_should_be": "what the correct answer/behavior should be"}

If is_correction is false, set what_was_wrong and what_should_be to empty strings."""


def detect_correction(
    conversation: list[dict[str, str]],
    llm,
) -> CorrectionResult:
    """Analyze conversation to detect if the latest user message is a correction."""
    if not conversation or conversation[-1]["role"] != "user":
        raise ValueError("Conversation must end with a user message")

    last_assistant_msg = ""
    for msg in reversed(conversation[:-1]):
        if msg["role"] == "assistant":
            last_assistant_msg = msg["content"]
            break

    analysis_messages = [
        {"role": "system", "content": DETECTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Here is the conversation to analyze:\n\n"
                f"{_format_conversation(conversation)}\n\n"
                "Is the user's last message a correction?"
            ),
        },
    ]

    raw_output = llm.generate(
        analysis_messages,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=False,
    )

    parsed = _parse_detection_output(raw_output)

    return CorrectionResult(
        is_correction=parsed["is_correction"],
        what_was_wrong=parsed["what_was_wrong"],
        what_should_be=parsed["what_should_be"],
        original_model_response=last_assistant_msg,
        raw_llm_output=raw_output,
    )


def _format_conversation(conversation: list[dict[str, str]]) -> str:
    lines = []
    for msg in conversation:
        role = msg["role"].upper()
        lines.append(f"{role}: {msg['content']}")
    return "\n\n".join(lines)


def _parse_detection_output(raw: str) -> dict:
    """Parse JSON output from detection LLM with fallbacks."""
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        return {
            "is_correction": bool(result.get("is_correction", False)),
            "what_was_wrong": str(result.get("what_was_wrong", "")),
            "what_should_be": str(result.get("what_should_be", "")),
        }
    except json.JSONDecodeError:
        match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                return {
                    "is_correction": bool(result.get("is_correction", False)),
                    "what_was_wrong": str(result.get("what_was_wrong", "")),
                    "what_should_be": str(result.get("what_should_be", "")),
                }
            except json.JSONDecodeError:
                pass
        return {"is_correction": False, "what_was_wrong": "", "what_should_be": ""}
