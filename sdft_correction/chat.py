"""Interactive chat loop that detects corrections and triggers SDFT training.

Pipeline:
  1. User chats with the local model
  2. Correction detector fires when user corrects the model  (API or local)
  3. Augmenter generates diverse prompt variations           (API or local)
  4. Local model is unloaded to free GPU
  5. GPT-4o-mini generates expert demonstrations             (API, no GPU)
  6. Data formatter builds the SDFT dataset
  7. DistilTrainer runs on-policy SDFT                       (GPU)
  8. Reuse trainer's vLLM for verification + continued chat

Set config.use_local_for_structured=True to use the local model for steps 2-3
instead of GPT-4o-mini. Works better with larger models (7B+).
"""
import os
from pathlib import Path

from sdft_correction.config import PipelineConfig
from sdft_correction.inference import LocalInference, OpenAIInference
from sdft_correction.correction_detector import detect_correction
from sdft_correction.augmenter import generate_prompt_variations
from sdft_correction.expert_demos import generate_expert_demos
from sdft_correction.data_formatter import format_for_sdft
from sdft_correction.trainer import run_sdft_training


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

    print("\n=== SDFT Correction Chat ===")
    print("Chat with the model. If you correct it, SDFT training will trigger.")
    print("Type 'quit' to exit, 'reset' to clear conversation.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            conversation = []
            print("[Conversation reset]\n")
            continue

        conversation.append({"role": "user", "content": user_input})

        # Check for correction (need at least user-assistant-user)
        if len(conversation) >= 3:
            print("[Checking for correction...]")
            result = detect_correction(conversation, smart_llm)

            if result.is_correction:
                print(f"\n[CORRECTION DETECTED]")
                print(f"  Wrong: {result.what_was_wrong}")
                print(f"  Should be: {result.what_should_be}")

                original_question = _find_original_question(conversation)

                # ── Phase 1: Augmentation (GPT-4o-mini API) ──
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

                # ── Phase 2: Free GPU ──
                print("[Unloading inference model...]")
                llm.unload()

                # ── Phase 3: Expert demos (API, no GPU needed) ──
                n_demos = len(all_prompts) * config.num_expert_demos_per_prompt
                print(f"[Generating {n_demos} expert demos via {config.openai_model}...]")
                expert_demo_list = generate_expert_demos(
                    prompts=all_prompts,
                    what_was_wrong=result.what_was_wrong,
                    what_should_be=result.what_should_be,
                    demos_per_prompt=config.num_expert_demos_per_prompt,
                    model=config.openai_model,
                )
                print(f"[Generated {len(expert_demo_list)} expert demonstrations]")
                for i, demo in enumerate(expert_demo_list):
                    print(f"  [DEMO {i+1}] Q: {demo.prompt[:80]}...")
                    print(f"           A: {demo.demonstration[:120]}...")

                # ── Phase 4: Format dataset ──
                dataset = format_for_sdft(expert_demo_list)
                print(f"[Dataset ready: {len(dataset)} training pairs]")

                # ── Phase 5: SDFT training (GPU) ──
                print("[Starting SDFT training...]")
                model_path, llm = run_sdft_training(
                    dataset=dataset,
                    model_name=config.model_name,
                    output_dir=str(config.output_dir),
                    learning_rate=config.learning_rate,
                    num_train_epochs=config.num_train_epochs,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    max_prompt_length=config.max_prompt_length,
                    max_completion_length=config.max_completion_length,
                )
                print(f"[Training complete. Model saved to {model_path}]")

                # Update model_name so the next correction trains from
                # this checkpoint (continual learning, not from scratch).
                config.model_name = model_path

                # ── Phase 6: Verification (reusing trainer's vLLM) ──

                print("\n[VERIFICATION - fresh context, analogous question]")
                verify_q = variations[0] if variations else original_question
                verify_response = llm.generate(
                    [{"role": "user", "content": verify_q}],
                    max_new_tokens=config.max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                )
                print(f"  Q: {verify_q}")
                print(f"  A: {verify_response}\n")

                conversation.append({"role": "assistant", "content": verify_response})
                continue

        # Normal response
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
