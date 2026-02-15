from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from sdft_platform.db import get_db
from sdft_platform.deps import AuthContext, get_optional_auth_context, require_auth
from sdft_platform.models import AuditEvent, User
from sdft_platform.schemas import AuthResponse, LoginRequest, RegisterRequest, UserOut
from sdft_platform.security import (
    clear_auth_cookies,
    clear_failed_logins,
    create_session,
    delete_session_by_raw_token,
    enforce_login_rate_limit,
    hash_password,
    record_failed_login,
    set_auth_cookies,
    validate_password_strength,
    verify_password,
)
from sdft_platform.settings import get_settings


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def register(payload: RegisterRequest, request: Request, response: Response, db: Session = Depends(get_db)) -> AuthResponse:
    settings = get_settings()
    email = payload.email.strip().lower()
    if "@" not in email or len(email) > 320:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email")
    validate_password_strength(payload.password)

    existing = db.scalar(select(User).where(User.email == email))
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        email=email,
        full_name=payload.full_name.strip(),
        password_hash=hash_password(payload.password),
        is_active=True,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    db.refresh(user)

    token, csrf_token = create_session(db, user, request, settings)
    set_auth_cookies(response, token, csrf_token, settings)

    db.add(AuditEvent(actor_user_id=user.id, action="auth_register", details={"email": email}))
    db.commit()

    return AuthResponse(user=UserOut.model_validate(user), csrf_token=csrf_token)


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)) -> AuthResponse:
    settings = get_settings()
    email = payload.email.strip().lower()
    enforce_login_rate_limit(email)

    user = db.scalar(select(User).where(User.email == email))
    if user is None or not verify_password(payload.password, user.password_hash):
        record_failed_login(email)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    clear_failed_logins(email)
    token, csrf_token = create_session(db, user, request, settings)
    set_auth_cookies(response, token, csrf_token, settings)

    db.add(AuditEvent(actor_user_id=user.id, action="auth_login", details={"email": email}))
    db.commit()

    return AuthResponse(user=UserOut.model_validate(user), csrf_token=csrf_token)


@router.post("/logout")
def logout(
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
    ctx: AuthContext = Depends(require_auth),
) -> dict[str, str]:
    settings = get_settings()
    raw_token = request.cookies.get(settings.session_cookie_name)
    if raw_token:
        delete_session_by_raw_token(db, raw_token)
    clear_auth_cookies(response, settings)

    db.add(AuditEvent(actor_user_id=ctx.user.id, action="auth_logout", details=None))
    db.commit()
    return {"status": "ok"}


@router.get("/me", response_model=AuthResponse)
def me(ctx: AuthContext | None = Depends(get_optional_auth_context)) -> AuthResponse:
    if ctx is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return AuthResponse(user=UserOut.model_validate(ctx.user), csrf_token=ctx.session.csrf_token)
