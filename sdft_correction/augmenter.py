"""Generates diverse prompt variations from a correction pattern.

The augmenter produces only *questions* — expert demonstrations for each
question are generated separately by expert_demos.py via GPT-4o-mini.
"""
import json
import re


AUGMENTATION_SYSTEM_PROMPT = """\
You are a training data generator. Given a correction a user made to an AI \
assistant, generate diverse prompts that would REQUIRE the corrected behavior.

CRITICAL: Generate prompts similar to what the ORIGINAL USER would say — \
requests, questions, or instructions that would trigger the same situation. \
Do NOT generate meta-questions ABOUT the correction (e.g. "Should X be done?" \
or "What format should be used for X?"). Instead generate prompts where the \
model must DEMONSTRATE the corrected behavior in its response.

Examples:
- If correction is "write poems in uppercase" → generate poem requests like \
"Write a poem about the ocean", "Can you compose a verse about winter?"
- If correction is "the year is 2026" → generate questions like \
"What year comes after 2025?", "What year is it right now?"

Each prompt should:
1. Be a natural user request/question similar to the original
2. Require the model to demonstrate the corrected behavior in its response
3. Vary in topic/phrasing from each other
4. Stay short and direct

Generate exactly {n} prompts. Output ONLY a JSON array of strings, no other text."""


def generate_prompt_variations(
    what_was_wrong: str,
    what_should_be: str,
    original_question: str,
    original_wrong_response: str,
    llm,
    n: int = 8,
) -> list[str]:
    """Generate *n* diverse prompt variations that test the same error pattern."""
    prompt_content = f"""Original situation:
- User asked: "{original_question}"
- Model incorrectly responded: "{original_wrong_response}"
- What was wrong: {what_was_wrong}
- Correct behavior: {what_should_be}

Generate {n} prompts that a user might naturally say, where the model would \
need to demonstrate the corrected behavior. Vary the topic but keep the same \
type of request as the original."""

    messages = [
        {"role": "system", "content": AUGMENTATION_SYSTEM_PROMPT.format(n=n)},
        {"role": "user", "content": prompt_content},
    ]

    raw_output = llm.generate(
        messages,
        max_new_tokens=2048,
        temperature=0.8,
        do_sample=True,
    )

    return _parse_variations(raw_output)


def _parse_variations(raw: str) -> list[str]:
    """Parse a JSON array of strings from LLM output, with fallbacks."""
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    try:
        items = json.loads(cleaned)
        if isinstance(items, list):
            return [str(item) for item in items if isinstance(item, str)]
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", raw)
        if match:
            try:
                items = json.loads(match.group())
                if isinstance(items, list):
                    return [str(item) for item in items if isinstance(item, str)]
            except json.JSONDecodeError:
                pass
    return []
