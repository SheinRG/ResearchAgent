"""
Authentication service — JWT token management, password hashing, Google OAuth validation.
Provides stateless auth for the FastAPI backend.
"""

import jwt
import bcrypt
import hashlib
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_token(user_id: str, email: str, name: str = "") -> str:
    """
    Create a JWT token for a user.

    Args:
        user_id: Unique user ID.
        email: User's email.
        name: User's display name.

    Returns:
        Signed JWT string.
    """
    settings = get_settings()
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=settings.auth_token_expiry_hours),
    }
    return jwt.encode(payload, settings.auth_secret, algorithm="HS256")


def validate_token(token: str) -> Optional[dict]:
    """
    Validate a JWT token and return the payload.

    Args:
        token: JWT string from Authorization header.

    Returns:
        Decoded payload dict, or None if invalid/expired.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.auth_secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid token: %s", e)
        return None


def _hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _utcnow() -> datetime:
    # Naive UTC, matching the DateTime columns in models/database.py.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def create_refresh_token() -> str:
    return secrets.token_urlsafe(32)


async def store_refresh_token(user_id: str, token: str, expiry_seconds: int) -> None:
    from app.models.database import RefreshToken, get_session_factory

    factory = get_session_factory()
    async with factory() as db:
        db.add(
            RefreshToken(
                token_hash=_hash_refresh_token(token),
                user_id=user_id,
                expires_at=_utcnow() + timedelta(seconds=expiry_seconds),
            )
        )
        await db.commit()


async def validate_and_rotate_refresh_token(token: str) -> Optional[tuple[str, str]]:
    """
    Look up the token, delete it, issue a new one (rotation).
    Returns (user_id, new_token) or None if the token is unknown or expired.
    Rotation means a stolen token is detected the next time the legitimate user refreshes.
    """
    from app.models.database import RefreshToken, get_session_factory

    factory = get_session_factory()
    async with factory() as db:
        row = await db.get(RefreshToken, _hash_refresh_token(token))
        if row is None:
            return None
        if row.expires_at <= _utcnow():
            await db.delete(row)
            await db.commit()
            return None
        user_id = row.user_id
        settings = get_settings()
        new_token = create_refresh_token()
        await db.delete(row)
        db.add(
            RefreshToken(
                token_hash=_hash_refresh_token(new_token),
                user_id=user_id,
                expires_at=_utcnow() + timedelta(days=settings.refresh_token_expiry_days),
            )
        )
        await db.commit()
        return user_id, new_token


async def revoke_refresh_token(token: str) -> None:
    from app.models.database import RefreshToken, get_session_factory

    factory = get_session_factory()
    async with factory() as db:
        row = await db.get(RefreshToken, _hash_refresh_token(token))
        if row is not None:
            await db.delete(row)
            await db.commit()


async def verify_google_token(credential: str) -> Optional[dict]:
    """
    Verify a Google OAuth ID token and return user info.

    Uses Google's tokeninfo endpoint to validate the credential.

    Args:
        credential: Google ID token string from frontend.

    Returns:
        Dict with email, name, google_id, picture — or None on failure.
    """
    settings = get_settings()

    # Fail closed: if no client ID is configured, Google login is disabled.
    # Without a client ID we cannot validate the token's audience, so accepting
    # tokens here would let an attacker replay a Google ID token minted for any
    # other OAuth client (an audience-confusion account takeover). Reject
    # outright instead of skipping the check.
    if not settings.google_client_id:
        logger.warning("Google auth attempted but GOOGLE_CLIENT_ID is not configured")
        return None

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={credential}"
            )
            if response.status_code != 200:
                logger.warning("Google token verification failed: %s", response.status_code)
                return None

            data = response.json()

            # Verify the token was issued for *our* client ID (audience).
            if data.get("aud") != settings.google_client_id:
                logger.warning("Google token audience mismatch")
                return None

            # Verify the token was issued by Google.
            if data.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
                logger.warning("Google token issuer invalid: %s", data.get("iss"))
                return None

            # Only trust verified emails — we link/create accounts by email, so an
            # unverified address would allow impersonating an arbitrary email.
            # tokeninfo returns this as the string "true"/"false".
            if str(data.get("email_verified", "")).lower() != "true":
                logger.warning("Google token email not verified")
                return None

            email = data.get("email", "")
            google_id = data.get("sub", "")
            if not email or not google_id:
                logger.warning("Google token missing email or subject")
                return None

            return {
                "email": email,
                "name": data.get("name", ""),
                "google_id": google_id,
                "picture": data.get("picture", ""),
            }
    except Exception as e:
        logger.error("Google token verification error: %s", e)
        return None
