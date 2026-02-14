"""Shared configuration for the sdft-correction pipeline."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    # Models
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    openai_model: str = "gpt-4o-mini"

    # When True, use the local model for correction detection and augmentation
    # instead of GPT-4o-mini. Works better with larger models (7B+).
    use_local_for_structured: bool = False

    # Paths — sdft_repo_path is the parent of this file's directory
    sdft_repo_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "output")

    # SDFT training (no LoRA — full fine-tuning, matching the paper)
    learning_rate: float = 1e-5
    num_train_epochs: int = 8
    gradient_accumulation_steps: int = 1  # effective batch = 4
    max_prompt_length: int = 512
    max_completion_length: int = 80

    # Augmentation & expert demos
    num_prompt_variations: int = 4
    num_expert_demos_per_prompt: int = 2
    # Total training pairs: (num_prompt_variations + 1) * num_expert_demos_per_prompt
    # = 5 * 2 = 10

    # Inference
    max_new_tokens: int = 150
    temperature: float = 0.7
