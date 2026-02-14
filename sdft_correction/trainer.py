"""Wraps the SDFT DistilTrainer for correction-based fine-tuning.

No LoRA — full fine-tuning, matching the paper's experimental setup.
The training loop (inside DistilTrainer) works as follows for each batch:

  1. Student generates on-policy rollout   y ~ pi_theta(. | prompt)
  2. Teacher computes logprobs             pi_EMA(y | teacher_prompt)
     where teacher_prompt = prompt + expert demo in-context
  3. Minimise  KL( pi_theta(. | prompt)  ||  pi_EMA(. | teacher_prompt) )
  4. Teacher weights updated via EMA of student weights
"""
import gc
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add Self-Distillation repo root to path so we can import distil_*
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from distil_config import DistilConfig  # noqa: E402
from distil_trainer import DistilTrainer  # noqa: E402


class TrainerInference:
    """Wraps the trainer's HF model for chat inference after training.

    Reuses the already-loaded model on GPU — no reloading from disk.
    """

    def __init__(self, trainer: DistilTrainer, tokenizer, save_path: str):
        self._trainer = trainer
        self.tokenizer = tokenizer
        self.device = "cuda"
        self._save_path = save_path
        self._saved = False
        self._trainer.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """Generate a completion using the trainer's HF model directly."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self._trainer.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                repetition_penalty=1.3,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def save(self):
        """Save the trained model to disk (called lazily before cleanup)."""
        if self._saved or self._trainer is None:
            return
        print(f"[Saving model to {self._save_path}...]")
        self._trainer.model.save_pretrained(self._save_path)
        self.tokenizer.save_pretrained(self._save_path)
        self._saved = True
        print(f"[Save complete]")

    def unload(self):
        """Clean up and free GPU memory. Call /save first if you want to persist."""
        if self._trainer is None:
            return
        if hasattr(self._trainer, 'ref_model'):
            del self._trainer.ref_model
        if hasattr(self._trainer, 'model'):
            del self._trainer.model
        del self._trainer
        self._trainer = None
        gc.collect()
        torch.cuda.empty_cache()


def run_sdft_training(
    dataset: Dataset,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str | None = None,
    learning_rate: float = 5e-5,
    num_train_epochs: int = 2,
    gradient_accumulation_steps: int = 8,
    max_prompt_length: int = 512,
    max_completion_length: int = 256,
) -> tuple[str, TrainerInference]:
    """Run SDFT training on a correction dataset.

    Returns (save_path, trainer_inference) — the saved model path and a
    TrainerInference wrapper for immediate inference without reloading.
    """
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent / "output")

    os.makedirs(output_dir, exist_ok=True)

    # ── Load student and teacher (same architecture, same init weights) ──
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── DistilConfig — no vLLM, HF generate for on-policy rollouts ──
    # For small training runs (< 20 samples), HF generate is faster than
    # vLLM due to vLLM's init + CUDA graph compilation overhead.
    config = DistilConfig(
        output_dir=output_dir,
        seed=42,
        use_vllm=False,
        # Optimiser
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        # Batch: effective batch = per_device * gradient_accumulation
        per_device_train_batch_size=4,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_train_epochs=num_train_epochs,
        save_strategy="no",
        save_steps=9999,
        max_grad_norm=1.0,
        report_to="none",
        log_completions=True,
        # SDFT-specific (on-policy, forward KL — matching reference main.py)
        num_generations=1,            # One rollout per prompt (distillation, not RL)
        generate_from_teacher=False,  # Student generates on-policy rollouts
        sync_ref_model=True,  # EMA teacher updates
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=0.01,  # EMA rate alpha
        num_loss_tokens_to_skip=3,  # Suppress teacher artifacts (Section 6)
    )

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Don't save now — TrainerInference.save() does it lazily on unload/quit
    save_path = os.path.join(output_dir, "trained_model")

    llm = TrainerInference(trainer, tokenizer, save_path)
    return save_path, llm
