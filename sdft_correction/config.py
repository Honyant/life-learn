"""Shared configuration for the sdft-correction pipeline."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    # Models
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    openai_model: str = "gpt-4o-mini"

    # Paths — sdft_repo_path is the parent of this file's directory
    sdft_repo_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "output")

    # SDFT training (no LoRA — full fine-tuning, matching the paper)
    learning_rate: float = 5e-5
    num_train_epochs: int = 2
    gradient_accumulation_steps: int = 8  # effective batch = 8
    max_prompt_length: int = 512
    max_completion_length: int = 256

    # Augmentation & expert demos
    num_prompt_variations: int = 8
    num_expert_demos_per_prompt: int = 4
    # Total training pairs: (num_prompt_variations + 1) * num_expert_demos_per_prompt
    # = 9 * 4 = 36  (small for 0.5B test; scale up for larger models)

    # Inference
    max_new_tokens: int = 512
    temperature: float = 0.7
