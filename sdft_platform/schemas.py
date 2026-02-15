from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from sdft_platform.models import MembershipRole, MessageRole, ModelVersionStatus, TrainingJobStatus


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    full_name: str


class RegisterRequest(BaseModel):
    email: str
    full_name: str = Field(min_length=2, max_length=120)
    password: str = Field(min_length=12, max_length=256)


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    user: UserOut
    csrf_token: str


class OrganizationCreateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    slug: str | None = Field(default=None, max_length=160)


class OrganizationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    created_at: datetime
    role: MembershipRole
    active_model_version_id: int | None


class MemberOut(BaseModel):
    user_id: int
    email: str
    full_name: str
    role: MembershipRole
    joined_at: datetime


class InviteCreateRequest(BaseModel):
    email: str
    role: MembershipRole = MembershipRole.member


class MemberRoleUpdateRequest(BaseModel):
    role: MembershipRole


class InviteOut(BaseModel):
    id: int
    email: str
    role: MembershipRole
    expires_at: datetime
    invite_token: str
    invite_url: str


class InviteAcceptRequest(BaseModel):
    token: str


class ChatThreadCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=240)


class ChatThreadOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    org_id: int
    title: str
    created_by_id: int
    created_at: datetime
    updated_at: datetime


class ChatMessageCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=12000)
    is_correction: bool = False
    trigger_training: bool = True


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    thread_id: int
    org_id: int
    role: MessageRole
    content: str
    is_correction: bool
    created_by_id: int | None
    created_at: datetime


class ModelVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    org_id: int
    version_number: int
    model_path: str
    parent_model_version_id: int | None
    status: ModelVersionStatus
    created_by_id: int
    notes: str | None
    created_at: datetime


class TrainingJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    org_id: int
    requested_by_id: int
    thread_id: int
    source_message_id: int
    model_version_id: int | None
    status: TrainingJobStatus
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ChatSendResponse(BaseModel):
    user_message: ChatMessageOut
    assistant_message: ChatMessageOut
    training_job: TrainingJobOut | None


class ManualTrainingRequest(BaseModel):
    thread_id: int
    source_message_id: int


class RollbackRequest(BaseModel):
    model_version_id: int


class OrgModelSnapshot(BaseModel):
    active_model_version: ModelVersionOut | None
    versions: list[ModelVersionOut]
