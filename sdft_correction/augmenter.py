"""Generates diverse prompt variations from a correction pattern.

The augmenter produces only *questions* — expert demonstrations for each
question are generated separately by expert_demos.py via GPT-4o-mini.
"""
import json
import re


AUGMENTATION_SYSTEM_PROMPT = """\
You are a training data generator. Given a correction a user made to an AI \
assistant, generate diverse question prompts that test the SAME specific \
knowledge or reasoning that was corrected.

Each question should:
1. Test the same specific fact, concept, or reasoning pattern that was corrected
2. Be phrased differently from the original and from each other — use \
rephrasings, follow-up angles, related sub-questions, or "what if" framings
3. Be a natural, standalone question a user might ask
4. Stay in the same or a closely related domain — do NOT jump to unrelated topics

Generate exactly {n} questions. Output ONLY a JSON array of strings, no other text."""


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

Generate {n} diverse questions in different domains/scenarios that test the same \
underlying reasoning or knowledge pattern."""

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
