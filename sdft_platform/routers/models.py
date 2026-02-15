from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from sdft_platform.db import get_db
from sdft_platform.deps import AuthContext, require_auth, require_org_role
from sdft_platform.models import (
    AuditEvent,
    ChatMessage,
    ChatThread,
    MembershipRole,
    ModelVersion,
    ModelVersionStatus,
    Organization,
    TrainingJob,
    TrainingJobStatus,
)
from sdft_platform.schemas import (
    ManualTrainingRequest,
    ModelVersionOut,
    OrgModelSnapshot,
    RollbackRequest,
    TrainingJobOut,
)
from sdft_platform.security import assert_csrf
from sdft_platform.services.model_runtime import OrgModelRuntime
from sdft_platform.services.training_service import TrainingCoordinator


router = APIRouter(prefix="/api", tags=["models"])


def _runtime(request: Request) -> OrgModelRuntime:
    return request.app.state.model_runtime


def _trainer(request: Request) -> TrainingCoordinator:
    return request.app.state.training_coordinator


@router.get("/orgs/{org_id}/model", response_model=OrgModelSnapshot)
def get_org_model(org_id: int, ctx: AuthContext = Depends(require_auth), db: Session = Depends(get_db)) -> OrgModelSnapshot:
    require_org_role(db, org_id, ctx.user.id, MembershipRole.member)
    org = db.get(Organization, org_id)
    if org is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    versions = list(
        db.scalars(
            select(ModelVersion)
            .where(ModelVersion.org_id == org_id)
            .order_by(ModelVersion.version_number.desc())
        )
    )
    active = db.get(ModelVersion, org.active_model_version_id) if org.active_model_version_id else None
    return OrgModelSnapshot(
        active_model_version=ModelVersionOut.model_validate(active) if active else None,
        versions=[ModelVersionOut.model_validate(version) for version in versions],
    )


@router.get("/orgs/{org_id}/jobs", response_model=list[TrainingJobOut])
def list_training_jobs(org_id: int, ctx: AuthContext = Depends(require_auth), db: Session = Depends(get_db)) -> list[TrainingJobOut]:
    require_org_role(db, org_id, ctx.user.id, MembershipRole.member)
    jobs = list(
        db.scalars(
            select(TrainingJob)
            .where(TrainingJob.org_id == org_id)
            .order_by(TrainingJob.created_at.desc(), TrainingJob.id.desc())
            .limit(25)
        )
    )
    return [TrainingJobOut.model_validate(job) for job in jobs]


@router.post("/orgs/{org_id}/train", response_model=TrainingJobOut)
def start_training_job(
    org_id: int,
    payload: ManualTrainingRequest,
    request: Request,
    ctx: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> TrainingJobOut:
    assert_csrf(request, ctx.session)
    require_org_role(db, org_id, ctx.user.id, MembershipRole.member)

    thread = db.get(ChatThread, payload.thread_id)
    if thread is None or thread.org_id != org_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found in this organization")

    source = db.get(ChatMessage, payload.source_message_id)
    if source is None or source.thread_id != thread.id or source.org_id != org_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source message not found")

    source.is_correction = True
    job = TrainingJob(
        org_id=org_id,
        requested_by_id=ctx.user.id,
        thread_id=thread.id,
        source_message_id=source.id,
        status=TrainingJobStatus.queued,
    )
    db.add(job)
    db.add(source)
    db.add(
        AuditEvent(
            org_id=org_id,
            actor_user_id=ctx.user.id,
            action="training_job_queued",
            details={"thread_id": thread.id, "source_message_id": source.id},
        )
    )
    db.commit()
    db.refresh(job)

    _trainer(request).enqueue(job.id)
    return TrainingJobOut.model_validate(job)


@router.post("/orgs/{org_id}/model/rollback", response_model=ModelVersionOut)
def rollback_model(
    org_id: int,
    payload: RollbackRequest,
    request: Request,
    ctx: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> ModelVersionOut:
    assert_csrf(request, ctx.session)
    require_org_role(db, org_id, ctx.user.id, MembershipRole.owner)

    org = db.get(Organization, org_id)
    if org is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    target = db.get(ModelVersion, payload.model_version_id)
    if target is None or target.org_id != org_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model version not found")
    if target.status == ModelVersionStatus.failed:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot rollback to a failed model version")

    previous_id = org.active_model_version_id
    org.active_model_version_id = target.id
    target.status = ModelVersionStatus.active
    db.add(target)

    if previous_id is not None and previous_id != target.id:
        previous = db.get(ModelVersion, previous_id)
        if previous is not None and previous.status != ModelVersionStatus.failed:
            previous.status = ModelVersionStatus.archived
            db.add(previous)

    db.add(org)
    db.add(
        AuditEvent(
            org_id=org_id,
            actor_user_id=ctx.user.id,
            action="model_rollback",
            details={
                "to_model_version_id": target.id,
                "from_model_version_id": previous_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
    )
    db.commit()

    _runtime(request).invalidate_org(org_id)
    return ModelVersionOut.model_validate(target)
