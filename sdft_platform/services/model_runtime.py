from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from sqlalchemy.orm import Session

from sdft_platform.models import Organization
from sdft_platform.settings import Settings


class OpenAIBackend:
    def __init__(self, model: str):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI()

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
        )
        return response.choices[0].message.content

    def unload(self) -> None:
        return


class MockInference:
    """Fallback inference backend for local development."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        last_user = ""
        for item in reversed(messages):
            if item["role"] == "user":
                last_user = item["content"]
                break
        if not last_user:
            return "I am ready. Ask me anything."
        return (
            f"[Mock:{self.model_name}] I heard: '{last_user}'. "
            "Set LL_INFERENCE_BACKEND=local (or openai) to use a real model response."
        )

    def unload(self) -> None:
        return


@dataclass
class RuntimeEntry:
    model_name: str
    inference: object


class OrgModelRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = RLock()
        self._entries: dict[int, RuntimeEntry] = {}

    def _effective_model_name(self, db: Session, org_id: int) -> str:
        org = db.get(Organization, org_id)
        if org is None:
            return self.settings.base_model_name
        if org.active_model_version is not None and org.active_model_version.model_path:
            candidate = org.active_model_version.model_path
            model_path = Path(candidate)
            if model_path.exists():
                return str(model_path)
            if "/" in candidate:
                return candidate
        return self.settings.base_model_name

    def _build_inference(self, model_name: str):
        backend = self.settings.inference_backend.strip().lower()
        if backend == "local":
            from sdft_correction.inference import LocalInference

            return LocalInference(model_name)
        if backend == "openai":
            return OpenAIBackend(model=self.settings.openai_model)
        return MockInference(model_name)

    def get_runtime(self, db: Session, org_id: int):
        model_name = self._effective_model_name(db, org_id)
        with self._lock:
            entry = self._entries.get(org_id)
            if entry is not None and entry.model_name == model_name:
                return entry.inference

            if entry is not None:
                try:
                    entry.inference.unload()
                except Exception:
                    pass

            try:
                inference = self._build_inference(model_name)
            except Exception:
                inference = MockInference(model_name)

            self._entries[org_id] = RuntimeEntry(model_name=model_name, inference=inference)
            return inference

    def generate(
        self,
        db: Session,
        org_id: int,
        messages: list[dict[str, str]],
        max_new_tokens: int = 220,
        temperature: float = 0.7,
    ) -> str:
        runtime = self.get_runtime(db, org_id)
        return runtime.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    def invalidate_org(self, org_id: int) -> None:
        with self._lock:
            entry = self._entries.pop(org_id, None)
        if entry is None:
            return
        try:
            entry.inference.unload()
        except Exception:
            pass

    def resolve_model_name(self, db: Session, org_id: int) -> str:
        return self._effective_model_name(db, org_id)

    def shutdown(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            try:
                entry.inference.unload()
            except Exception:
                pass
