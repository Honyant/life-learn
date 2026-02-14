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


def run_sdft_training(
    dataset: Dataset,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str | None = None,
    learning_rate: float = 5e-5,
    num_train_epochs: int = 2,
    gradient_accumulation_steps: int = 8,
    max_prompt_length: int = 512,
    max_completion_length: int = 256,
) -> str:
    """Run SDFT training on a correction dataset.

    Returns path to the saved model/adapter directory.
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

    # ── DistilConfig — mirrors main.py in the Self-Distillation repo ──
    config = DistilConfig(
        output_dir=output_dir,
        seed=42,
        # vLLM for on-policy student generation
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.6,
        vllm_enable_sleep_mode=True,
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
        save_steps=9999,
        max_grad_norm=1.0,
        report_to="none",
        log_completions=True,
        # SDFT-specific (on-policy, forward KL — matching reference main.py)
        # alpha=0.0 is forward KL; alpha=1.0 is reverse KL.
        # The reference implementation uses the default (0.0 = forward KL).
        generate_from_teacher=False,  # Student generates on-policy rollouts
        sync_ref_model=True,  # EMA teacher updates
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=0.01,  # EMA rate alpha
        vllm_importance_sampling_correction=True,  # Correct vLLM/train mismatch
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

    # Save the trained model
    save_path = os.path.join(output_dir, "trained_model")
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Cleanup — vLLM sleep mode allocates a persistent GPU memory pool that
    # must be explicitly released before a new vLLM instance can be created
    # in the same process, otherwise: "Sleep mode can only be used for one
    # instance per process."
    if hasattr(trainer, 'llm') and trainer.llm is not None:
        llm_engine = getattr(trainer.llm, 'llm_engine', None)
        if llm_engine is not None and hasattr(llm_engine, 'shutdown'):
            llm_engine.shutdown()
        del trainer.llm
    del trainer, model, teacher_model
    gc.collect()
    torch.cuda.empty_cache()

    return save_path
