from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from sdft_platform.db import get_db
from sdft_platform.models import MembershipRole, OrganizationMembership, User, UserSession
from sdft_platform.security import get_session_by_raw_token
from sdft_platform.settings import get_settings


ROLE_RANK = {
    MembershipRole.member: 1,
    MembershipRole.admin: 2,
    MembershipRole.owner: 3,
}


@dataclass
class AuthContext:
    user: User
    session: UserSession


def get_optional_auth_context(request: Request, db: Session = Depends(get_db)) -> AuthContext | None:
    settings = get_settings()
    raw_token = request.cookies.get(settings.session_cookie_name)
    if not raw_token:
        return None
    session = get_session_by_raw_token(db, raw_token)
    if session is None:
        return None
    user = db.get(User, session.user_id)
    if user is None or not user.is_active:
        return None
    return AuthContext(user=user, session=session)


def require_auth(ctx: AuthContext | None = Depends(get_optional_auth_context)) -> AuthContext:
    if ctx is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return ctx


def require_org_role(
    db: Session,
    org_id: int,
    user_id: int,
    minimum_role: MembershipRole = MembershipRole.member,
) -> OrganizationMembership:
    stmt = select(OrganizationMembership).where(
        OrganizationMembership.org_id == org_id,
        OrganizationMembership.user_id == user_id,
    )
    membership = db.scalar(stmt)
    if membership is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of this organization")
    if ROLE_RANK[membership.role] < ROLE_RANK[minimum_role]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role for this action")
    return membership
