from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("LL_APP_NAME", "Life-Learn Workspace")
    secret_key: str = os.getenv("LL_SECRET_KEY", "dev-secret-change-me")
    cookie_secure: bool = _as_bool(os.getenv("LL_COOKIE_SECURE"), default=False)
    session_cookie_name: str = os.getenv("LL_SESSION_COOKIE", "ll_session")
    csrf_cookie_name: str = os.getenv("LL_CSRF_COOKIE", "ll_csrf")
    session_ttl_hours: int = int(os.getenv("LL_SESSION_TTL_HOURS", "168"))

    database_url: str = os.getenv(
        "LL_DATABASE_URL",
        f"sqlite:///{(Path(__file__).resolve().parent.parent / 'life_learn.db').as_posix()}",
    )

    base_model_name: str = os.getenv("LL_BASE_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    openai_model: str = os.getenv("LL_OPENAI_MODEL", "gpt-4o-mini")
    inference_backend: str = os.getenv("LL_INFERENCE_BACKEND", "mock")
    use_local_for_structured: bool = _as_bool(os.getenv("LL_USE_LOCAL_STRUCTURED"), default=False)

    model_storage_dir: Path = Path(
        os.getenv(
            "LL_MODEL_STORAGE_DIR",
            str(Path(__file__).resolve().parent.parent / "sdft_correction" / "output" / "org_models"),
        )
    )

    train_learning_rate: float = float(os.getenv("LL_TRAIN_LR", "1e-5"))
    train_num_epochs: int = int(os.getenv("LL_TRAIN_EPOCHS", "8"))
    train_gradient_accumulation_steps: int = int(os.getenv("LL_TRAIN_GRAD_ACC", "1"))
    train_max_prompt_length: int = int(os.getenv("LL_TRAIN_MAX_PROMPT_LEN", "512"))
    train_max_completion_length: int = int(os.getenv("LL_TRAIN_MAX_COMPLETION_LEN", "80"))
    num_prompt_variations: int = int(os.getenv("LL_PROMPT_VARIATIONS", "4"))
    num_expert_demos_per_prompt: int = int(os.getenv("LL_EXPERT_DEMOS_PER_PROMPT", "2"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
