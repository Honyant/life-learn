from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from sdft_platform.db import get_db
from sdft_platform.deps import AuthContext, require_auth, require_org_role
from sdft_platform.models import (
    AuditEvent,
    MembershipRole,
    ModelVersion,
    ModelVersionStatus,
    Organization,
    OrganizationInvite,
    OrganizationMembership,
    User,
)
from sdft_platform.schemas import (
    InviteAcceptRequest,
    InviteCreateRequest,
    InviteOut,
    MemberOut,
    MemberRoleUpdateRequest,
    OrganizationCreateRequest,
    OrganizationOut,
)
from sdft_platform.security import assert_csrf
from sdft_platform.settings import get_settings


router = APIRouter(prefix="/api", tags=["organizations"])


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s-]", "", name).strip().lower()
    cleaned = re.sub(r"[\s_-]+", "-", cleaned)
    cleaned = cleaned.strip("-")
    return cleaned[:160] or "org"


def _next_available_slug(db: Session, base_slug: str) -> str:
    slug = base_slug
    suffix = 2
    while db.scalar(select(func.count()).select_from(Organization).where(Organization.slug == slug)):
        slug = f"{base_slug}-{suffix}"
        suffix += 1
    return slug


def _membership_query(user_id: int) -> Select[tuple[OrganizationMembership]]:
    return select(OrganizationMembership).where(OrganizationMembership.user_id == user_id)


@router.get("/orgs", response_model=list[OrganizationOut])
def list_organizations(ctx: AuthContext = Depends(require_auth), db: Session = Depends(get_db)) -> list[OrganizationOut]:
    memberships = list(db.scalars(_membership_query(ctx.user.id)))
    result: list[OrganizationOut] = []
    for membership in memberships:
        org = db.get(Organization, membership.org_id)
        if org is None:
            continue
        result.append(
            OrganizationOut(
                id=org.id,
                name=org.name,
                slug=org.slug,
                created_at=org.created_at,
                role=membership.role,
                active_model_version_id=org.active_model_version_id,
            )
        )
    result.sort(key=lambda item: item.name.lower())
    return result


@router.post("/orgs", response_model=OrganizationOut)
def create_organization(
    payload: OrganizationCreateRequest,
    request: Request,
    ctx: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> OrganizationOut:
    assert_csrf(request, ctx.session)
    settings = get_settings()

    base_slug = payload.slug.strip().lower() if payload.slug else _slugify(payload.name)
    if not re.fullmatch(r"[a-z0-9-]{2,160}", base_slug):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Slug must contain only lowercase letters, numbers, and hyphens",
        )
    slug = _next_available_slug(db, base_slug)

    org = Organization(name=payload.name.strip(), slug=slug, created_by_id=ctx.user.id)
    db.add(org)
    db.flush()

    owner_membership = OrganizationMembership(org_id=org.id, user_id=ctx.user.id, role=MembershipRole.owner)
    db.add(owner_membership)

    bootstrap_model = ModelVersion(
        org_id=org.id,
        version_number=1,
        model_path=settings.base_model_name,
        parent_model_version_id=None,
        status=ModelVersionStatus.active,
        created_by_id=ctx.user.id,
        notes="Organization bootstrap base model",
    )
    db.add(bootstrap_model)
    db.flush()

    org.active_model_version_id = bootstrap_model.id
    db.add(org)
    db.add(
        AuditEvent(
            org_id=org.id,
            actor_user_id=ctx.user.id,
            action="organization_created",
            details={"slug": slug, "name": payload.name.strip()},
        )
    )
    db.commit()

    return OrganizationOut(
        id=org.id,
        name=org.name,
        slug=org.slug,
        created_at=org.created_at,
        role=MembershipRole.owner,
        active_model_version_id=org.active_model_version_id,
    )


@router.get("/orgs/{org_id}/members", response_model=list[MemberOut])
def list_members(org_id: int, ctx: AuthContext = Depends(require_auth), db: Session = Depends(get_db)) -> list[MemberOut]:
    require_org_role(db, org_id, ctx.user.id, MembershipRole.member)

    stmt = (
        select(OrganizationMembership, User)
        .join(User, User.id == OrganizationMembership.user_id)
        .where(OrganizationMembership.org_id == org_id)
        .order_by(OrganizationMembership.created_at.asc())
    )
    rows = db.execute(stmt).all()
    return [
        MemberOut(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=membership.role,
            joined_at=membership.created_at,
        )
        for membership, user in rows
    ]


@router.post("/orgs/{org_id}/invites", response_model=InviteOut)
def create_invite(
    org_id: int,
    payload: InviteCreateRequest,
    request: Request,
    ctx: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> InviteOut:
    assert_csrf(request, ctx.session)
    require_org_role(db, org_id, ctx.user.id, MembershipRole.admin)

    email = payload.email.strip().lower()
    if "@" not in email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid invite email")

    token = secrets.token_urlsafe(36)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expires_at = datetime.utcnow() + timedelta(days=7)
    invite = OrganizationInvite(
        org_id=org_id,
        email=email,
        role=payload.role,
        token_hash=token_hash,
        invited_by_id=ctx.user.id,
        expires_at=expires_at,
    )
    db.add(invite)
    db.add(
        AuditEvent(
            org_id=org_id,
            actor_user_id=ctx.user.id,
            action="organization_invite_created",
            details={"email": email, "role": payload.role.value},
        )
    )
    db.commit()
    db.refresh(invite)

    base_url = str(request.base_url).rstrip("/")
    invite_url = f"{base_url}/invite/{token}"
    return InviteOut(
        id=invite.id,
        email=invite.email,
        role=invite.role,
        expires_at=invite.expires_at,
        invite_token=token,
        invite_url=invite_url,
    )


@router.post("/invites/accept", response_model=OrganizationOut)
def accept_invite(
    payload: InviteAcceptRequest,
    request: Request,
    ctx: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> OrganizationOut:
    assert_csrf(request, ctx.session)
    token_hash = hashlib.sha256(payload.token.encode("utf-8")).hexdigest()

    stmt = select(OrganizationInvite).where(OrganizationInvite.token_hash == token_hash)
    invite = db.scalar(stmt)
    if invite is None or invite.accepted_at is not None or invite.expires_at < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invite token is invalid or expired")

    if ctx.user.email.lower() != invite.email.lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invite email does not match your account email",
        )

    membership = db.scalar(
        select(OrganizationMembership).where(
            OrganizationMembership.org_id == invite.org_id,
            OrganizationMembership.user_id == ctx.user.id,
        )
    )
    if membership is None:
        membership = OrganizationMembership(org_id=invite.org_id, user_id=ctx.user.id, role=invite.role)
        db.add(membership)
    else:
        membership.role = invite.role
        db.add(membership)

    invite.accepted_at = datetime.utcnow()
    db.add(invite)
    db.add(
        AuditEvent(
            org_id=invite.org_id,
            actor_user_id=ctx.user.id,
            action="organization_invite_accepted",
            details={"invite_id": invite.id, "role": invite.role.value},
        )
    )
    db.commit()

    org = db.get(Organization, invite.org_id)
    if org is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    return OrganizationOut(
        id=org.id,
        name=org.name,
        slug=org.slug,
        created_at=org.created_at,
        role=membership.role,
        active_model_version_id=org.active_model_version_id,
    )


@router.patch("/orgs/{org_id}/members/{member_user_id}", response_model=MemberOut)
def update_member_role(
    org_id: int,
    member_user_id: int,
    payload: MemberRoleUpdateRequest,
    request: Request,
    ctx: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> MemberOut:
    assert_csrf(request, ctx.session)
    require_org_role(db, org_id, ctx.user.id, MembershipRole.owner)

    membership = db.scalar(
        select(OrganizationMembership).where(
            OrganizationMembership.org_id == org_id,
            OrganizationMembership.user_id == member_user_id,
        )
    )
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")

    membership.role = payload.role
    db.add(membership)
    db.add(
        AuditEvent(
            org_id=org_id,
            actor_user_id=ctx.user.id,
            action="organization_member_role_updated",
            details={"target_user_id": member_user_id, "new_role": payload.role.value},
        )
    )
    db.commit()

    user = db.get(User, member_user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return MemberOut(
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=membership.role,
        joined_at=membership.created_at,
    )


@router.delete("/orgs/{org_id}/members/{member_user_id}")
def remove_member(
    org_id: int,
    member_user_id: int,
    request: Request,
    ctx: AuthContext = Depends(require_auth),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    assert_csrf(request, ctx.session)
    actor_membership = require_org_role(db, org_id, ctx.user.id, MembershipRole.admin)

    membership = db.scalar(
        select(OrganizationMembership).where(
            OrganizationMembership.org_id == org_id,
            OrganizationMembership.user_id == member_user_id,
        )
    )
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")

    if membership.role == MembershipRole.owner and actor_membership.role != MembershipRole.owner:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only owners can remove owners")

    if membership.role == MembershipRole.owner:
        owner_count = db.scalar(
            select(func.count())
            .select_from(OrganizationMembership)
            .where(OrganizationMembership.org_id == org_id, OrganizationMembership.role == MembershipRole.owner)
        )
        if owner_count <= 1:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Organization must retain at least one owner")

    db.delete(membership)
    db.add(
        AuditEvent(
            org_id=org_id,
            actor_user_id=ctx.user.id,
            action="organization_member_removed",
            details={"target_user_id": member_user_id},
        )
    )
    db.commit()
    return {"status": "ok"}
