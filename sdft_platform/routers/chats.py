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
    MessageRole,
    TrainingJob,
    TrainingJobStatus,
)
from sdft_platform.schemas import (
    ChatMessageCreateRequest,
    ChatMessageOut,
    ChatSendResponse,
    ChatThreadCreateRequest,
    ChatThreadOut,
    TrainingJobOut,
)
from sdft_platform.security import assert_csrf
from sdft_platform.services.model_runtime import OrgModelRuntime
from sdft_platform.services.training_service import TrainingCoordinator


router = APIRouter(prefix="/api", tags=["chat"])


def _runtime(request: Request) -> OrgModelRuntime:
    return request.app.state.model_runtime


def _trainer(request: Request) -> TrainingCoordinator:
    return request.app.state.training_coordinator


@router.get("/orgs/{org_id}/threads", response_model=list[ChatThreadOut])
def list_threads(org_id: int, ctx: AuthContext = Depends(require_auth), db: Session = Depends(get_db)) -> list[ChatThreadOut]:
    require_org_role(db, org_id, ctx.user.id, MembershipRole.member)
    stmt = (
        select(ChatThread)
        .where(ChatThread.org_id == org_id)
        .order_by(ChatThread.updated_at.desc(), ChatThread.id.desc())
    )
    return [ChatThreadOut.model_validate(thread) for thread in db.scalars(stmt)]


@router.post("/orgs/{org_id}/threads", response_model=ChatThreadOut)
def create_thread(
    org_id: int,
    payload: ChatThreadCreateRequest,
    request: Request,
    ctx: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> ChatThreadOut:
    assert_csrf(request, ctx.session)
    require_org_role(db, org_id, ctx.user.id, MembershipRole.member)

    now = datetime.utcnow()
    thread = ChatThread(
        org_id=org_id,
        title=payload.title.strip(),
        created_by_id=ctx.user.id,
        created_at=now,
        updated_at=now,
    )
    db.add(thread)
    db.add(
        AuditEvent(
            org_id=org_id,
            actor_user_id=ctx.user.id,
            action="thread_created",
            details={"thread_title": payload.title.strip()},
        )
    )
    db.commit()
    db.refresh(thread)
    return ChatThreadOut.model_validate(thread)


@router.get("/threads/{thread_id}/messages", response_model=list[ChatMessageOut])
def list_messages(thread_id: int, ctx: AuthContext = Depends(require_auth), db: Session = Depends(get_db)) -> list[ChatMessageOut]:
    thread = db.get(ChatThread, thread_id)
    if thread is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    require_org_role(db, thread.org_id, ctx.user.id, MembershipRole.member)

    stmt = select(ChatMessage).where(ChatMessage.thread_id == thread_id).order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
    return [ChatMessageOut.model_validate(message) for message in db.scalars(stmt)]


@router.post("/threads/{thread_id}/messages", response_model=ChatSendResponse)
def send_message(
    thread_id: int,
    payload: ChatMessageCreateRequest,
    request: Request,
    ctx: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> ChatSendResponse:
    assert_csrf(request, ctx.session)

    thread = db.get(ChatThread, thread_id)
    if thread is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    require_org_role(db, thread.org_id, ctx.user.id, MembershipRole.member)

    existing_messages = list(
        db.scalars(
            select(ChatMessage)
            .where(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        )
    )
    conversation = [{"role": msg.role.value, "content": msg.content} for msg in existing_messages]

    now = datetime.utcnow()
    user_msg = ChatMessage(
        thread_id=thread.id,
        org_id=thread.org_id,
        role=MessageRole.user,
        content=payload.content.strip(),
        is_correction=payload.is_correction,
        created_by_id=ctx.user.id,
        created_at=now,
    )
    db.add(user_msg)
    db.flush()

    conversation.append({"role": "user", "content": user_msg.content})
    runtime = _runtime(request)
    assistant_text = runtime.generate(
        db,
        thread.org_id,
        conversation,
        max_new_tokens=220,
        temperature=0.7,
    )

    assistant_msg = ChatMessage(
        thread_id=thread.id,
        org_id=thread.org_id,
        role=MessageRole.assistant,
        content=assistant_text,
        is_correction=False,
        created_by_id=None,
        created_at=datetime.utcnow(),
    )
    db.add(assistant_msg)

    thread.updated_at = datetime.utcnow()
    db.add(thread)

    training_job = None
    if payload.is_correction and payload.trigger_training:
        training_job = TrainingJob(
            org_id=thread.org_id,
            requested_by_id=ctx.user.id,
            thread_id=thread.id,
            source_message_id=user_msg.id,
            status=TrainingJobStatus.queued,
        )
        db.add(training_job)

    db.add(
        AuditEvent(
            org_id=thread.org_id,
            actor_user_id=ctx.user.id,
            action="chat_message_sent",
            details={
                "thread_id": thread.id,
                "is_correction": payload.is_correction,
                "training_enqueued": bool(training_job),
            },
        )
    )

    db.commit()
    db.refresh(user_msg)
    db.refresh(assistant_msg)
    if training_job is not None:
        db.refresh(training_job)
        _trainer(request).enqueue(training_job.id)

    return ChatSendResponse(
        user_message=ChatMessageOut.model_validate(user_msg),
        assistant_message=ChatMessageOut.model_validate(assistant_msg),
        training_job=TrainingJobOut.model_validate(training_job) if training_job else None,
    )
