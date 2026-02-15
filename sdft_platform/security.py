from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from datetime import datetime, timedelta

from fastapi import HTTPException, Request, Response, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from sdft_platform.models import User, UserSession
from sdft_platform.settings import Settings


try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError

    _password_hasher = PasswordHasher()
    _has_argon2 = True
except ImportError:
    _password_hasher = None
    _has_argon2 = False


_attempt_lock = threading.Lock()
_failed_attempts: dict[str, list[float]] = {}
_max_attempts = 8
_window_seconds = 300


def hash_password(password: str) -> str:
    if _has_argon2:
        return f"argon2${_password_hasher.hash(password)}"

    iterations = 390_000
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    if password_hash.startswith("argon2$") and _has_argon2:
        try:
            return _password_hasher.verify(password_hash[len("argon2$") :], password)
        except VerifyMismatchError:
            return False

    if password_hash.startswith("pbkdf2_sha256$"):
        try:
            _, iter_str, salt_hex, digest_hex = password_hash.split("$", 3)
            iterations = int(iter_str)
            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(digest_hex)
            candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
            return hmac.compare_digest(candidate, expected)
        except Exception:
            return False

    if _has_argon2:
        try:
            return _password_hasher.verify(password_hash, password)
        except VerifyMismatchError:
            return False
    return False


def validate_password_strength(password: str) -> None:
    if len(password) < 12:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must be at least 12 characters")
    if password.lower() == password or password.upper() == password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must include both uppercase and lowercase characters",
        )
    if not any(ch.isdigit() for ch in password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must include at least one number")


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(db: Session, user: User, request: Request, settings: Settings) -> tuple[str, str]:
    raw_token = secrets.token_urlsafe(48)
    csrf_token = secrets.token_urlsafe(24)
    now = datetime.utcnow()
    expires_at = now + timedelta(hours=settings.session_ttl_hours)

    session = UserSession(
        user_id=user.id,
        token_hash=hash_session_token(raw_token),
        csrf_token=csrf_token,
        expires_at=expires_at,
        created_at=now,
        last_seen_at=now,
        user_agent=(request.headers.get("user-agent") or "")[:255],
        ip_address=(request.client.host if request.client else None),
    )
    db.add(session)
    db.commit()
    return raw_token, csrf_token


def get_session_by_raw_token(db: Session, raw_token: str) -> UserSession | None:
    token_hash = hash_session_token(raw_token)
    stmt = select(UserSession).where(UserSession.token_hash == token_hash)
    session = db.scalar(stmt)
    if session is None:
        return None
    if session.expires_at < datetime.utcnow():
        db.delete(session)
        db.commit()
        return None
    return session


def delete_session_by_raw_token(db: Session, raw_token: str) -> None:
    token_hash = hash_session_token(raw_token)
    db.execute(delete(UserSession).where(UserSession.token_hash == token_hash))
    db.commit()


def touch_session(db: Session, session: UserSession) -> None:
    session.last_seen_at = datetime.utcnow()
    db.commit()


def assert_csrf(request: Request, session: UserSession) -> None:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    token = request.headers.get("x-csrf-token")
    if token is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Missing CSRF token")
    if not hmac.compare_digest(token, session.csrf_token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")


def set_auth_cookies(response: Response, session_token: str, csrf_token: str, settings: Settings) -> None:
    max_age = settings.session_ttl_hours * 3600
    response.set_cookie(
        key=settings.session_cookie_name,
        value=session_token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        max_age=max_age,
        path="/",
    )
    response.set_cookie(
        key=settings.csrf_cookie_name,
        value=csrf_token,
        httponly=False,
        secure=settings.cookie_secure,
        samesite="lax",
        max_age=max_age,
        path="/",
    )


def clear_auth_cookies(response: Response, settings: Settings) -> None:
    response.delete_cookie(settings.session_cookie_name, path="/")
    response.delete_cookie(settings.csrf_cookie_name, path="/")


def enforce_login_rate_limit(email: str) -> None:
    key = email.strip().lower()
    now = time.time()
    with _attempt_lock:
        attempts = _failed_attempts.setdefault(key, [])
        attempts[:] = [ts for ts in attempts if now - ts <= _window_seconds]
        if len(attempts) >= _max_attempts:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed login attempts. Please wait a few minutes.",
            )


def record_failed_login(email: str) -> None:
    key = email.strip().lower()
    with _attempt_lock:
        attempts = _failed_attempts.setdefault(key, [])
        attempts.append(time.time())


def clear_failed_logins(email: str) -> None:
    key = email.strip().lower()
    with _attempt_lock:
        _failed_attempts.pop(key, None)
