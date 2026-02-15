"""Interactive chat loop that detects corrections and triggers SDFT training.

Pipeline:
  1. User chats with the local model
  2. Correction detector fires when user corrects the model  (API or local)
  3. Augmenter generates diverse prompt variations           (API or local)
  4. Local model is unloaded to free GPU
  5. Training runs in a BACKGROUND THREAD on GPU
  6. While training: chat is served by hijacking the training model
     between optimizer steps (via ChatCallback)
  7. When training completes: seamlessly switch to TrainerInference

Set config.use_local_for_structured=True to use the local model for steps 2-3
instead of GPT-4o-mini. Works better with larger models (7B+).
"""
import logging
import os
import threading
import traceback
from pathlib import Path

from sdft_correction.config import PipelineConfig
from sdft_correction.inference import LocalInference, OpenAIInference
from sdft_correction.correction_detector import detect_correction
from sdft_correction.augmenter import generate_prompt_variations
from sdft_correction.expert_demos import generate_expert_demos
from sdft_correction.data_formatter import format_for_sdft
from sdft_correction.trainer import ChatCallback, run_sdft_training


def _run_training_in_background(
    all_prompts,
    what_was_wrong,
    what_should_be,
    config,
    existing_model,
    existing_tokenizer,
    chat_callback,
    result_holder,
):
    """Background thread: expert demos → format → train."""
    # Suppress noisy loggers so they don't pollute the chat terminal
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    try:
        # ── Expert demos (API, no GPU needed) ──
        n_demos = len(all_prompts) * config.num_expert_demos_per_prompt
        print(f"[BG] Generating {n_demos} expert demos via {config.openai_model}...")
        expert_demo_list = generate_expert_demos(
            prompts=all_prompts,
            what_was_wrong=what_was_wrong,
            what_should_be=what_should_be,
            demos_per_prompt=config.num_expert_demos_per_prompt,
            model=config.openai_model,
        )
        print(f"[BG] Generated {len(expert_demo_list)} expert demonstrations")

        # ── Format dataset ──
        dataset = format_for_sdft(expert_demo_list)
        print(f"[BG] Dataset ready: {len(dataset)} training pairs")

        # ── SDFT training (GPU, quiet) — chat served via callback between steps ──
        print("[BG] Starting SDFT training (quiet mode)...")
        model_path, trained_llm = run_sdft_training(
            dataset=dataset,
            model_name=config.model_name,
            output_dir=str(config.output_dir),
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_prompt_length=config.max_prompt_length,
            max_completion_length=config.max_completion_length,
            existing_model=existing_model,
            existing_tokenizer=existing_tokenizer,
            chat_callback=chat_callback,
            quiet=True,
        )

        # ── Verification ──
        verify_q = all_prompts[1] if len(all_prompts) > 1 else all_prompts[0]
        verify_response = trained_llm.generate(
            [{"role": "user", "content": verify_q}],
            max_new_tokens=config.max_new_tokens,
            temperature=0.1,
            do_sample=False,
        )
        print(f"\n[Training complete!]")
        print(f"  Verification Q: {verify_q}")
        print(f"  Verification A: {verify_response}\n")

        result_holder["llm"] = trained_llm

    except Exception as e:
        print(f"\n[BG] Training failed: {e}")
        traceback.print_exc()
        result_holder["error"] = str(e)


def run_chat_pipeline(config: PipelineConfig | None = None):
    """Main interactive chat loop."""
    if config is None:
        config = PipelineConfig()

    # Load the .env file if it exists (for OPENAI_API_KEY)
    _load_env(config)

    # Check if a previously trained model exists — resume from it
    trained_path = config.output_dir / "trained_model"
    if trained_path.is_dir() and (trained_path / "config.json").exists():
        print(f"[Found trained model at {trained_path}, resuming from checkpoint]")
        config.model_name = str(trained_path)
    else:
        print(f"[No trained model found, starting from {config.model_name}]")

    print("Loading model for chat...")
    llm = LocalInference(config.model_name)

    # Structured tasks (correction detection, augmentation) can use either
    # GPT-4o-mini (reliable JSON) or the local model (no API dependency).
    if config.use_local_for_structured:
        print("[Using LOCAL model for correction detection & augmentation]")
        smart_llm = llm
    else:
        print("[Using OpenAI API for correction detection & augmentation]")
        smart_llm = OpenAIInference(model=config.openai_model)

    conversation: list[dict[str, str]] = []

    # Background training state
    training_thread = None
    training_result = {}

    print("\n=== SDFT Correction Chat ===")
    print("Chat with the model. If you correct it, SDFT training will trigger.")
    print("Type 'quit' to exit, 'reset' to clear conversation, '/save' to save model.\n")

    while True:
        # ── Check if background training finished ──
        if training_thread is not None and not training_thread.is_alive():
            training_thread = None
            if "llm" in training_result:
                llm = training_result["llm"]
                if config.use_local_for_structured:
                    smart_llm = llm
            elif "error" in training_result:
                print("[Training failed — continuing with current model]\n")
            training_result = {}

        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            if training_thread is not None:
                print("[Waiting for background training to finish...]")
                training_thread.join()
            break
        if user_input.lower() == "/save":
            if hasattr(llm, 'save'):
                llm.save()
            else:
                print("[No trained model to save]")
            continue
        if user_input.lower() == "reset":
            conversation = []
            print("[Conversation reset]\n")
            continue

        conversation.append({"role": "user", "content": user_input})

        # Check for correction (need at least user-assistant-user)
        # Skip correction detection if training is already running
        if len(conversation) >= 3 and training_thread is None:
            print("[Checking for correction...]")
            result = detect_correction(conversation, smart_llm)

            if result.is_correction:
                print(f"\n[CORRECTION DETECTED]")
                print(f"  Wrong: {result.what_was_wrong}")
                print(f"  Should be: {result.what_should_be}")

                original_question = _find_original_question(conversation)

                # ── Phase 1: Augmentation (stays in main thread — fast) ──
                print(f"\n[Generating {config.num_prompt_variations} prompt variations...]")
                variations = generate_prompt_variations(
                    what_was_wrong=result.what_was_wrong,
                    what_should_be=result.what_should_be,
                    original_question=original_question,
                    original_wrong_response=result.original_model_response,
                    llm=smart_llm,
                    n=config.num_prompt_variations,
                )
                all_prompts = [original_question] + variations
                print(f"[Got {len(all_prompts)} prompts]")
                for i, p in enumerate(all_prompts):
                    tag = "ORIGINAL" if i == 0 else f"VAR {i}"
                    print(f"  [{tag}] {p}")

                # ── Phase 2: Free GPU (keep model for continual learning) ──
                existing_model, existing_tokenizer = None, None
                if hasattr(llm, 'extract_model_and_tokenizer'):
                    existing_model, existing_tokenizer = llm.extract_model_and_tokenizer()
                # Grab tokenizer before unloading (LocalInference has .tokenizer)
                tokenizer = existing_tokenizer or getattr(llm, 'tokenizer', None)
                print("[Unloading inference model...]")
                llm.unload()

                # ── Phase 3: Create callback and launch background training ──
                # The callback hijacks the training model for chat between steps.
                chat_cb = ChatCallback(
                    tokenizer=tokenizer,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )
                # Chat goes through the callback — blocks until next step boundary
                llm = chat_cb
                if config.use_local_for_structured:
                    smart_llm = llm

                training_result = {}
                training_thread = threading.Thread(
                    target=_run_training_in_background,
                    args=(
                        all_prompts,
                        result.what_was_wrong,
                        result.what_should_be,
                        config,
                        existing_model,
                        existing_tokenizer,
                        chat_cb,
                        training_result,
                    ),
                    daemon=True,
                )
                training_thread.start()
                print("[Training in background — you can keep chatting!]")
                print("[Responses wait for next training step, ~5-15s]\n")

                conversation = []
                continue

        # ── Generate response ──
        # If the callback returned None, training just finished — re-check
        # for the trained model before responding.
        response = llm.generate(
            conversation,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
        if response is None:
            # Training finished between the user typing and us calling generate.
            # Swap to the trained model and retry.
            if training_thread is not None:
                training_thread.join()
                training_thread = None
            if "llm" in training_result:
                llm = training_result["llm"]
                if config.use_local_for_structured:
                    smart_llm = llm
                print("[Switched to trained model]\n")
            training_result = {}
            response = llm.generate(
                conversation,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )

        conversation.append({"role": "assistant", "content": response})
        print(f"Model: {response}\n")


# ──────────────────────────── helpers ────────────────────────────


def _find_original_question(conversation: list[dict]) -> str:
    """Find the user question that preceded the (wrong) assistant response."""
    if len(conversation) >= 3:
        return conversation[-3]["content"]
    return conversation[0]["content"]



def _load_env(config: PipelineConfig):
    """Try to load OPENAI_API_KEY from a .env file next to the project."""
    candidates = [
        Path.home() / "Documents" / "temp_dir" / ".env",
        config.sdft_repo_path / ".env",
        config.sdft_repo_path / "sdft_correction" / ".env",
    ]
    for env_path in candidates:
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip().removeprefix("export").strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key, val)
            break


if __name__ == "__main__":
    run_chat_pipeline()
