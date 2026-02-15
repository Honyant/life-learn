# Life-Learn: SDFT Chat Correction Pipeline

A backend that demonstrates **Self-Distillation Fine-Tuning (SDFT)** triggered by user corrections in chat. When a user corrects the model, the system detects the correction, generates expert demonstrations, runs on-policy SDFT, and the model permanently learns from the correction.

Based on the paper ["Self-Distillation Enables Continual Learning"](https://arxiv.org/abs/2601.19897) by Shenfeld et al.

## How It Works

```
User corrects the model in chat
    -> Correction Detector (local LLM classifier)
    -> Prompt Augmenter (local LLM generates diverse question variations)
    -> Expert Demo Generator (GPT-4o-mini produces correct demonstrations)
    -> Data Formatter (HuggingFace Dataset with prompt + teacher_prompt)
    -> SDFT Trainer (on-policy distillation via DistilTrainer)
    -> Verification (fresh context, analogous question -- model should get it right)
```

During SDFT training (handled internally by DistilTrainer):
- **Student** generates on-policy rollouts from the prompt alone
- **Teacher** (same model, EMA weights) is conditioned on prompt + expert demo
- Loss: KL divergence between student and teacher distributions
- Teacher weights track student via exponential moving average

## Setup

### 1. Clone

```bash
git clone https://github.com/Honyant/life-learn.git
cd life-learn
```

### 2. Create conda environment

```bash
conda env create -f environment.yml
conda activate sdft
```

Or manually:

```bash
conda create -n sdft python=3.10
conda activate sdft
pip install -r requirements.txt
pip install pytest openai
```

### 3. Set OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

Or place it in `~/Documents/temp_dir/.env`:
```
export OPENAI_API_KEY="sk-..."
```

## Usage

### Multi-tenant Web Workspace (new)

This repo now includes a full-stack workspace with:

- secure authentication (session cookies + CSRF)
- multi-organization tenancy with role-based access
- invite-based member management per org
- multi-chat threads per organization
- per-organization model versioning and rollback
- correction-triggered continual learning jobs (SDFT)

Run it with:

```bash
uvicorn sdft_platform.main:app --host 0.0.0.0 --port 8080 --reload
```

Then open `http://localhost:8080`.

Environment variables (optional):

| Variable | Default | Description |
|---|---|---|
| `LL_DATABASE_URL` | `sqlite:///life_learn.db` | Database connection string |
| `LL_SECRET_KEY` | `dev-secret-change-me` | App secret (set a strong value in production) |
| `LL_INFERENCE_BACKEND` | `mock` | `mock`, `local`, or `openai` |
| `LL_BASE_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Base model ID/path for new orgs |
| `LL_OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model for API-based inference/structured tasks |
| `LL_USE_LOCAL_STRUCTURED` | `false` | Use local model for correction detection/augmentation |
| `LL_MODEL_STORAGE_DIR` | `sdft_correction/output/org_models` | Per-org trained model storage root |

### Run unit tests (no GPU needed)

```bash
python -m pytest sdft_correction/test_pipeline.py -v -k 'not gpu'
```

### Run full integration test (GPU + API key)

```bash
python -m pytest sdft_correction/test_pipeline.py -v -m gpu -s
```

### Interactive demo

```bash
python -m sdft_correction.chat
```

Chat with the model. When you correct it, the pipeline triggers automatically:
1. Detects the correction
2. Generates 8 prompt variations via the local model
3. Gets 4 expert demos per prompt from GPT-4o-mini (~36 training pairs)
4. Runs SDFT training
5. Loads the trained model and verifies on an analogous question

## Project Structure

```
.
├── distil_config.py          # SDFT config (patched: added full_logit_distillation field)
├── distil_trainer.py         # DistilTrainer from the Self-Distillation paper
├── main.py                   # Original paper's training script (reference)
├── sdft_platform/            # New multi-tenant web workspace (FastAPI + UI)
├── environment.yml           # Conda environment file
├── requirements.txt          # Pip dependencies
└── sdft_correction/          # Chat correction pipeline
    ├── config.py             # PipelineConfig (model, paths, hyperparams)
    ├── inference.py          # Local model wrapper for generation
    ├── correction_detector.py # LLM-based correction classifier
    ├── augmenter.py          # Generates diverse prompt variations
    ├── expert_demos.py       # GPT-4o-mini expert demonstration generator
    ├── data_formatter.py     # Formats data for DistilTrainer
    ├── trainer.py            # SDFT training wrapper (full fine-tuning)
    ├── chat.py               # Interactive chat loop
    ├── conftest.py           # Pytest config
    └── test_pipeline.py      # 22 unit tests + 1 GPU integration test
```

## Configuration

Edit `sdft_correction/config.py` to change defaults:

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | Base model (use 7B+ for real results) |
| `openai_model` | `gpt-4o-mini` | Expert demo generator |
| `learning_rate` | `5e-5` | SDFT learning rate |
| `num_train_epochs` | `2` | Training epochs |
| `gradient_accumulation_steps` | `8` | Effective batch size |
| `num_prompt_variations` | `8` | Augmented prompts per correction |
| `num_expert_demos_per_prompt` | `4` | Expert demos per prompt |

## Key Design Decisions

- **On-policy student rollouts**: `generate_from_teacher=False` -- the student generates its own completions during training, matching the paper's Algorithm 1
- **External expert demos**: GPT-4o-mini provides the demonstration context `c` for the teacher, analogous to how the paper uses GPT-4o for SciKnowEval
- **Full fine-tuning**: No LoRA, matching the paper's experimental setup. Use larger GPUs for 7B+ models.
- **Forward KL** (`alpha=0.0`): Matches the reference `main.py` implementation
- **EMA teacher** (`sync_ref_model=True`, `ref_model_mixup_alpha=0.01`): Teacher tracks student progress while smoothing updates

## Original Paper

This project builds on the Self-Distillation repo by [Shenfeld et al.](https://github.com/idanshen/Self-Distillation):

> **Self-Distillation Enables Continual Learning** (ICLR 2025)
> Idan Shenfeld, Mehul Damani, Jonas Hubotter, Pulkit Agrawal
> https://arxiv.org/abs/2601.19897
