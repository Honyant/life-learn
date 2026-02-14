"""Formats expert demonstrations into the SDFT training dataset format.

The dataset has two columns consumed by DistilTrainer:
  - ``prompt``         — what the *student* sees  (just the question)
  - ``teacher_prompt`` — what the *teacher* sees   (question + expert demo in-context)

During training the student generates an on-policy rollout from ``prompt``,
and the KL divergence to the teacher (conditioned on ``teacher_prompt``) is
minimised.
"""
from string import Template

from datasets import Dataset

# Matches the template in Self-Distillation/main.py:30-37 and paper Section 3.
TEACHER_PROMPT_TEMPLATE = Template(
    """$question

This is an example for a response to the question:
$demonstration

Now answer with a short, direct response of your own."""
)


def format_for_sdft(expert_demos: list) -> Dataset:
    """Convert a list of :class:`ExpertDemo` objects into a HuggingFace Dataset.

    Each row is one (student prompt, teacher prompt) pair.
    """
    prompts = []
    teacher_prompts = []

    for demo in expert_demos:
        teacher_content = TEACHER_PROMPT_TEMPLATE.substitute(
            question=demo.prompt,
            demonstration=demo.demonstration,
        )

        prompts.append([{"role": "user", "content": demo.prompt}])
        teacher_prompts.append([{"role": "user", "content": teacher_content}])

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "teacher_prompt": teacher_prompts,
        }
    )
