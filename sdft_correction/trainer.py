"""Wraps the SDFT DistilTrainer for correction-based fine-tuning.

No LoRA — full fine-tuning, matching the paper's experimental setup.
The training loop (inside DistilTrainer) works as follows for each batch:

  1. Student generates on-policy rollout   y ~ pi_theta(. | prompt)
  2. Teacher computes logprobs             pi_EMA(y | teacher_prompt)
     where teacher_prompt = prompt + expert demo in-context
  3. Minimise  KL( pi_theta(. | prompt)  ||  pi_EMA(. | teacher_prompt) )
  4. Teacher weights updated via EMA of student weights
"""
import copy
import gc
import os
import queue
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

# Add Self-Distillation repo root to path so we can import distil_*
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from distil_config import DistilConfig  # noqa: E402
from distil_trainer import DistilTrainer  # noqa: E402


class ChatCallback(TrainerCallback):
    """HF Trainer callback that serves chat inference between training steps.

    The model is already on GPU for training.  Between optimizer steps
    (on_step_end), we check a request queue, flip to eval mode, generate
    a response, flip back to train mode, and put the response on a
    response queue.  The main thread can call .generate() which blocks
    until the next step boundary.
    """

    def __init__(self, tokenizer, max_new_tokens=150, temperature=0.7):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._request_q = queue.Queue()
        self._response_q = queue.Queue()
        self._model = None
        self._training_done = False
        # Progress tracking for web UI
        self.current_step = 0
        self.total_steps = 0
        self.training_phase = "idle"

    # ── TrainerCallback hooks ──

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._model = model
        self.total_steps = state.max_steps
        self.current_step = 0
        self.training_phase = "training"

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._model = model
        self.current_step = state.global_step
        self._drain_requests()

    def on_train_end(self, args, state, control, **kwargs):
        self._training_done = True
        self.training_phase = "complete"
        # Unblock anyone waiting — they'll get None and know training ended
        self._drain_requests()

    # ── Public interface (called from main thread) ──

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool = True,
    ) -> str:
        """Send a chat request and block until the next step boundary."""
        if self._training_done:
            return None
        self._request_q.put({
            "messages": messages,
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": temperature or self.temperature,
            "do_sample": do_sample,
        })
        # Block until the callback processes it (or training ends)
        resp = self._response_q.get(timeout=300)
        return resp

    def unload(self):
        """No-op — model lifecycle is managed by the trainer."""
        pass

    # ── Internal ──

    def _drain_requests(self):
        """Process all pending chat requests using the training model."""
        while not self._request_q.empty():
            try:
                req = self._request_q.get_nowait()
            except queue.Empty:
                break

            if self._model is None or self._training_done:
                self._response_q.put(None)
                continue

            model = self._model
            was_training = model.training
            model.eval()

            try:
                text = self.tokenizer.apply_chat_template(
                    req["messages"], tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=req["max_new_tokens"],
                        temperature=req["temperature"] if req["do_sample"] else 1.0,
                        do_sample=req["do_sample"],
                        repetition_penalty=1.3,
                    )
                new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            except Exception as e:
                response = f"[inference error: {e}]"
            finally:
                if was_training:
                    model.train()

            self._response_q.put(response)


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

    def extract_model_and_tokenizer(self):
        """Return (model, tokenizer) and detach them so unload() won't destroy them.

        Used for continual learning: pass the model to the next training run.
        """
        model = self._trainer.model
        tokenizer = self.tokenizer
        # Detach so unload doesn't delete the model
        self._trainer.model = None
        return model, tokenizer

    def unload(self):
        """Clean up and free GPU memory. Call /save first if you want to persist."""
        if self._trainer is None:
            return
        if hasattr(self._trainer, 'ref_model') and self._trainer.ref_model is not None:
            del self._trainer.ref_model
        if hasattr(self._trainer, 'model') and self._trainer.model is not None:
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
    existing_model=None,
    existing_tokenizer=None,
    chat_callback: ChatCallback | None = None,
    quiet: bool = False,
) -> tuple[str, TrainerInference]:
    """Run SDFT training on a correction dataset.

    Pass existing_model/existing_tokenizer for continual learning (avoids
    saving/loading from disk between corrections).

    Pass chat_callback to enable chat inference between training steps
    (for background training while the user keeps chatting).

    Set quiet=True to suppress tqdm, completion logs, and training logs
    (used when training runs in the background).

    Returns (save_path, trainer_inference) — the saved model path and a
    TrainerInference wrapper for immediate inference without reloading.
    """
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent / "output")

    os.makedirs(output_dir, exist_ok=True)

    # ── Load student and teacher (same architecture, same init weights) ──
    if existing_model is not None:
        if not quiet:
            print("[Continual learning: reusing in-memory model]")
        model = existing_model
        model.train()
        # Deep copy for teacher — same weights, separate parameters
        teacher_model = copy.deepcopy(existing_model)
        tokenizer = existing_tokenizer or AutoTokenizer.from_pretrained(model_name)
    else:
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
    #
    # steps_per_generation defaults to gradient_accumulation_steps, which
    # sets generation_batch_size = per_device_bs * steps_per_generation.
    # The RepeatSampler drops incomplete batches, so generation_batch_size
    # must be <= len(dataset).  Force steps_per_generation=1 so it equals
    # per_device_bs (4), which always fits small correction datasets.
    per_device_bs = 4

    config = DistilConfig(
        output_dir=output_dir,
        seed=42,
        use_vllm=False,
        # Optimiser
        learning_rate=learning_rate,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        logging_steps=1,
        bf16=True,
        fp16=False,
        # Batch: effective batch = per_device * gradient_accumulation
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        steps_per_generation=1,  # Generate every micro-batch (avoids empty sampler on small datasets)
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_train_epochs=num_train_epochs,
        save_strategy="no",
        save_steps=9999,
        max_grad_norm=1.0,
        report_to="none",
        # Quiet mode: suppress tqdm + completion tables when running in background
        log_completions=not quiet,
        disable_tqdm=quiet,
        logging_strategy="no" if quiet else "steps",
        # SDFT-specific (on-policy, forward KL — matching reference main.py)
        num_generations=1,            # One rollout per prompt (distillation, not RL)
        generate_from_teacher=False,  # Student generates on-policy rollouts
        sync_ref_model=False,  # Frozen teacher — preserve in-context demo advantage
        num_loss_tokens_to_skip=0,  # Was 3, but kills loss for short completions like "No."
    )

    callbacks = [chat_callback] if chat_callback else None

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()

    # Don't save now — TrainerInference.save() does it lazily on unload/quit
    save_path = os.path.join(output_dir, "trained_model")

    llm = TrainerInference(trainer, tokenizer, save_path)
    return save_path, llm
