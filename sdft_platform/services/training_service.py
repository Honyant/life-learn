from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from sdft_platform.models import (
    AuditEvent,
    ChatMessage,
    MessageRole,
    ModelVersion,
    ModelVersionStatus,
    Organization,
    TrainingJob,
    TrainingJobStatus,
)
from sdft_platform.services.model_runtime import OpenAIBackend, OrgModelRuntime
from sdft_platform.settings import Settings


class TrainingCoordinator:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker,
        model_runtime: OrgModelRuntime,
        max_workers: int = 2,
    ):
        self.settings = settings
        self.session_factory = session_factory
        self.model_runtime = model_runtime
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="train")
        self._org_locks: dict[int, Lock] = {}
        self._org_locks_guard = Lock()

    def _org_lock(self, org_id: int) -> Lock:
        with self._org_locks_guard:
            lock = self._org_locks.get(org_id)
            if lock is None:
                lock = Lock()
                self._org_locks[org_id] = lock
            return lock

    def enqueue(self, job_id: int) -> None:
        self.executor.submit(self._run_job, job_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)

    def _run_job(self, job_id: int) -> None:
        org_id = None
        with self.session_factory() as db:
            job = db.get(TrainingJob, job_id)
            if job is None:
                return
            org_id = job.org_id

        if org_id is None:
            return

        lock = self._org_lock(org_id)
        with lock:
            try:
                self._run_job_locked(job_id)
            except Exception as exc:
                error = f"{exc}\n{traceback.format_exc()}"
                self._mark_job_failed(job_id, error)

    def _run_job_locked(self, job_id: int) -> None:
        from sdft_correction.augmenter import generate_prompt_variations
        from sdft_correction.correction_detector import detect_correction
        from sdft_correction.data_formatter import format_for_sdft
        from sdft_correction.expert_demos import generate_expert_demos
        from sdft_correction.trainer import run_sdft_training

        with self.session_factory() as db:
            job = db.get(TrainingJob, job_id)
            if job is None:
                return
            if job.status not in {TrainingJobStatus.queued, TrainingJobStatus.running}:
                return

            job.status = TrainingJobStatus.running
            job.started_at = datetime.utcnow()
            db.add(job)
            db.commit()

            org = db.get(Organization, job.org_id)
            if org is None:
                raise ValueError("Organization not found")

            messages_stmt = (
                select(ChatMessage)
                .where(ChatMessage.thread_id == job.thread_id)
                .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
            )
            messages = list(db.scalars(messages_stmt))
            source = db.get(ChatMessage, job.source_message_id)
            if source is None:
                raise ValueError("Correction source message no longer exists")
            if source.role != MessageRole.user:
                raise ValueError("Training source message must be a user correction")

            conversation = [{"role": msg.role.value, "content": msg.content} for msg in messages]
            if not conversation or conversation[-1]["role"] != "user":
                # Train from context ending at correction message.
                cutoff = None
                for idx, msg in enumerate(messages):
                    if msg.id == source.id:
                        cutoff = idx
                        break
                if cutoff is None:
                    raise ValueError("Could not locate correction message in thread")
                conversation = [
                    {"role": msg.role.value, "content": msg.content}
                    for msg in messages[: cutoff + 1]
                ]

            if self.settings.use_local_for_structured and self.settings.inference_backend == "local":
                structured_llm = self.model_runtime.get_runtime(db, org.id)
            else:
                structured_llm = OpenAIBackend(model=self.settings.openai_model)

            detection = detect_correction(conversation, structured_llm)
            if not detection.is_correction:
                raise ValueError("Message was not detected as a correction; no training performed")

            original_question = self._find_original_question(conversation)
            variations = generate_prompt_variations(
                what_was_wrong=detection.what_was_wrong,
                what_should_be=detection.what_should_be,
                original_question=original_question,
                original_wrong_response=detection.original_model_response,
                llm=structured_llm,
                n=self.settings.num_prompt_variations,
            )
            prompts = [original_question] + variations

            expert_demos = generate_expert_demos(
                prompts=prompts,
                what_was_wrong=detection.what_was_wrong,
                what_should_be=detection.what_should_be,
                demos_per_prompt=self.settings.num_expert_demos_per_prompt,
                model=self.settings.openai_model,
            )
            if not expert_demos:
                raise ValueError("Failed to generate expert demonstrations")

            dataset = format_for_sdft(expert_demos)

            next_version = self._next_version_number(db, org.id)
            org_dir = self.settings.model_storage_dir / f"org_{org.id}"
            version_dir = org_dir / f"v{next_version}"
            version_dir.mkdir(parents=True, exist_ok=True)
            save_path = version_dir / "trained_model"

            new_model_version = ModelVersion(
                org_id=org.id,
                version_number=next_version,
                model_path=str(save_path),
                parent_model_version_id=org.active_model_version_id,
                status=ModelVersionStatus.training,
                created_by_id=job.requested_by_id,
                notes=f"Correction-driven fine-tune from thread {job.thread_id}",
            )
            db.add(new_model_version)
            db.flush()
            job.model_version_id = new_model_version.id
            db.add(job)
            db.commit()

            base_model_name = self.model_runtime.resolve_model_name(db, org.id)
            train_output_dir = str(version_dir)

            _, trained_runtime = run_sdft_training(
                dataset=dataset,
                model_name=base_model_name,
                output_dir=train_output_dir,
                learning_rate=self.settings.train_learning_rate,
                num_train_epochs=self.settings.train_num_epochs,
                gradient_accumulation_steps=self.settings.train_gradient_accumulation_steps,
                max_prompt_length=self.settings.train_max_prompt_length,
                max_completion_length=self.settings.train_max_completion_length,
            )
            trained_runtime.save()
            trained_runtime.unload()

            previous_active_id = org.active_model_version_id
            org.active_model_version_id = new_model_version.id
            new_model_version.status = ModelVersionStatus.active

            if previous_active_id is not None:
                prev = db.get(ModelVersion, previous_active_id)
                if prev is not None and prev.id != new_model_version.id:
                    prev.status = ModelVersionStatus.archived
                    db.add(prev)

            job.status = TrainingJobStatus.succeeded
            job.finished_at = datetime.utcnow()

            db.add_all([org, new_model_version, job])
            db.add(
                AuditEvent(
                    org_id=org.id,
                    actor_user_id=job.requested_by_id,
                    action="training_job_succeeded",
                    details={
                        "job_id": job.id,
                        "new_model_version_id": new_model_version.id,
                        "base_model": base_model_name,
                        "dataset_size": len(dataset),
                    },
                )
            )
            db.commit()

            self.model_runtime.invalidate_org(org.id)

    def _mark_job_failed(self, job_id: int, error: str) -> None:
        trimmed_error = error[-6000:]
        with self.session_factory() as db:
            job = db.get(TrainingJob, job_id)
            if job is None:
                return
            job.status = TrainingJobStatus.failed
            job.error_message = trimmed_error
            job.finished_at = datetime.utcnow()

            if job.model_version_id is not None:
                model_version = db.get(ModelVersion, job.model_version_id)
                if model_version is not None and model_version.status == ModelVersionStatus.training:
                    model_version.status = ModelVersionStatus.failed
                    db.add(model_version)

            db.add(job)
            db.add(
                AuditEvent(
                    org_id=job.org_id,
                    actor_user_id=job.requested_by_id,
                    action="training_job_failed",
                    details={"job_id": job.id, "error": trimmed_error[-1200:]},
                )
            )
            db.commit()

    @staticmethod
    def _next_version_number(db: Session, org_id: int) -> int:
        stmt = select(func.max(ModelVersion.version_number)).where(ModelVersion.org_id == org_id)
        max_version = db.scalar(stmt)
        return (max_version or 0) + 1

    @staticmethod
    def _find_original_question(conversation: list[dict[str, str]]) -> str:
        if len(conversation) >= 3:
            return conversation[-3]["content"]
        if conversation:
            return conversation[0]["content"]
        return ""
