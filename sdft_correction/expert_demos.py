"""Generates expert demonstrations using GPT-4o-mini.

These demonstrations serve as the context *c* in the paper's teacher formulation:
    teacher = pi(y | x, c)
where *c* is a correct expert response that the teacher model conditions on
via in-context learning.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class ExpertDemo:
    prompt: str
    demonstration: str


EXPERT_SYSTEM_PROMPT = """\
You are an expert assistant. Be concise — answer in 2-4 sentences max.

Context about an error pattern to avoid:
- Common mistake: {what_was_wrong}
- Correct approach: {what_should_be}

Give a short, accurate response to the user's question. Avoid the mistake \
pattern above. Be direct — no filler, no caveats."""


def generate_expert_demos(
    prompts: list[str],
    what_was_wrong: str,
    what_should_be: str,
    demos_per_prompt: int = 4,
    model: str = "gpt-4o-mini",
    max_workers: int = 16,
) -> list[ExpertDemo]:
    """Generate expert demonstrations for each prompt using GPT-4o-mini.

    Returns a list of ExpertDemo(prompt, demonstration) pairs — one per
    (prompt, demo_index) combination.
    """
    client = OpenAI()
    system_content = EXPERT_SYSTEM_PROMPT.format(
        what_was_wrong=what_was_wrong,
        what_should_be=what_should_be,
    )

    # Build flat task list: each entry is a prompt string
    tasks: list[str] = []
    for prompt in prompts:
        for _ in range(demos_per_prompt):
            tasks.append(prompt)

    results: list[ExpertDemo] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for prompt in tasks:
            future = executor.submit(
                _generate_single_demo, client, model, system_content, prompt
            )
            futures[future] = prompt

        for future in as_completed(futures):
            prompt = futures[future]
            try:
                demo_text = future.result()
                results.append(ExpertDemo(prompt=prompt, demonstration=demo_text))
            except Exception as e:
                print(f"Warning: Failed to generate demo: {e}")

    return results


def _generate_single_demo(
    client: OpenAI,
    model: str,
    system_content: str,
    prompt: str,
) -> str:
    """Generate a single expert demonstration via the OpenAI API."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.8,
    )
    return response.choices[0].message.content
