"""
Auth router — register, login, Google OAuth, /me endpoint.
Also exports get_current_user and check_rate_limit for use by other routers.
"""

import time
import uuid
import logging

from fastapi import APIRouter, HTTPException, Depends, Request, Response
from sqlalchemy import select

from app.config import get_settings
from app.models.schemas import (
    RegisterRequest,
    LoginRequest,
    GoogleAuthRequest,
    AuthResponse,
    ProfileUpdateRequest,
)
from app.models.database import User, get_session_factory
from app.services.auth import (
    hash_password,
    verify_password,
    create_token,
    validate_token,
    verify_google_token,
    create_refresh_token,
    store_refresh_token,
    validate_and_rotate_refresh_token,
    revoke_refresh_token,
)
from app.services.cache import get_redis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _refresh_expiry() -> int:
    return get_settings().refresh_token_expiry_days * 86400


def _set_refresh_cookie(response: Response, token: str) -> None:
    settings = get_settings()
    secure = settings.environment == "production"
    response.set_cookie(
        key="refresh_token",
        value=token,
        max_age=_refresh_expiry(),
        httponly=True,
        secure=secure,
        samesite="none" if secure else "lax",
        path="/api/auth",
    )


def _clear_refresh_cookie(response: Response) -> None:
    settings = get_settings()
    secure = settings.environment == "production"
    response.delete_cookie(
        key="refresh_token",
        path="/api/auth",
        secure=secure,
        samesite="none" if secure else "lax",
    )


async def get_current_user(request: Request) -> dict:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = auth_header[7:]
    payload = validate_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


_RATE_WINDOW_SECONDS = 3600

# Process-local fallback counters used only when Redis is unavailable, so a
# Redis outage bounds abuse per-instance instead of removing the limit entirely
# (each research query spends real money on Groq + Serper). Maps user_id ->
# (window_start_epoch, count). Not shared across instances — acceptable as a
# degraded mode; the single Render instance behaves the same as with Redis.
_local_buckets: dict[str, tuple[float, int]] = {}


def _rate_limit_exceeded(limit: int) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=f"Rate limit exceeded. Maximum {limit} queries per hour.",
    )


def _check_local_rate_limit(user_id: str, limit: int) -> None:
    """Fixed-window limiter in process memory (Redis-down fallback)."""
    now = time.time()

    # Opportunistically prune expired buckets so memory can't grow unbounded.
    if len(_local_buckets) > 10_000:
        for uid, (start, _) in list(_local_buckets.items()):
            if now - start >= _RATE_WINDOW_SECONDS:
                _local_buckets.pop(uid, None)

    window_start, count = _local_buckets.get(user_id, (now, 0))
    if now - window_start >= _RATE_WINDOW_SECONDS:
        window_start, count = now, 0  # window rolled over

    count += 1
    _local_buckets[user_id] = (window_start, count)
    if count > limit:
        raise _rate_limit_exceeded(limit)


async def check_rate_limit(user_id: str) -> None:
    """
    Fixed-window per-user rate limit. Atomic in Redis (INCR-then-check, so
    concurrent requests can't slip past the cap), with a process-local fallback
    when Redis is unavailable so the limit fails closed instead of wide open.
    """
    settings = get_settings()
    limit = settings.rate_limit_per_hour
    key = f"ratelimit:{user_id}"

    redis = await get_redis()
    if redis is None:
        _check_local_rate_limit(user_id, limit)
        return

    try:
        # INCR first, then check the returned value — this is atomic, unlike a
        # GET-then-INCR which races under concurrency. Set the TTL only on the
        # first hit of the window (nx) so abuse can't keep pushing it forward.
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, _RATE_WINDOW_SECONDS, nx=True)
        results = await pipe.execute()
        count = int(results[0])
        if count > limit:
            raise _rate_limit_exceeded(limit)
    except HTTPException:
        raise
    except Exception as e:
        # Redis errored mid-flight — degrade to the local limiter rather than
        # letting the request through uncounted.
        logger.warning("Rate limit (redis) failed, using local fallback: %s", e)
        _check_local_rate_limit(user_id, limit)


@router.post("/register")
async def register(request: RegisterRequest, response: Response):
    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(select(User).where(User.email == request.email))
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=409, detail="Registration failed. Please try again or log in.")
        user = User(
            id=str(uuid.uuid4()),
            email=request.email,
            name=request.name or request.email.split("@")[0],
            password_hash=hash_password(request.password),
        )
        db.add(user)
        await db.commit()
        token = create_token(user.id, user.email, user.name)
        refresh_tok = create_refresh_token()
        await store_refresh_token(user.id, refresh_tok, _refresh_expiry())
        _set_refresh_cookie(response, refresh_tok)
        logger.info("New user registered: %s", user.email)
        return AuthResponse(token=token, user={"id": user.id, "email": user.email, "name": user.name, "preferred_name": user.preferred_name or ""})


@router.post("/login")
async def login(request: LoginRequest, response: Response):
    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(select(User).where(User.email == request.email))
        user = result.scalar_one_or_none()
        if not user or not user.password_hash:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not verify_password(request.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_token(user.id, user.email, user.name)
        refresh_tok = create_refresh_token()
        await store_refresh_token(user.id, refresh_tok, _refresh_expiry())
        _set_refresh_cookie(response, refresh_tok)
        logger.info("User logged in: %s", user.email)
        return AuthResponse(token=token, user={"id": user.id, "email": user.email, "name": user.name, "preferred_name": user.preferred_name or ""})


@router.post("/google")
async def google_auth(request: GoogleAuthRequest, response: Response):
    google_info = await verify_google_token(request.credential)
    if not google_info:
        raise HTTPException(status_code=401, detail="Invalid Google credential")
    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(
            select(User).where(
                (User.google_id == google_info["google_id"]) | (User.email == google_info["email"])
            )
        )
        user = result.scalar_one_or_none()
        if user:
            if not user.google_id:
                user.google_id = google_info["google_id"]
            if google_info.get("picture"):
                user.picture = google_info["picture"]
            if google_info.get("name") and not user.name:
                user.name = google_info["name"]
            await db.commit()
        else:
            user = User(
                id=str(uuid.uuid4()),
                email=google_info["email"],
                name=google_info.get("name", google_info["email"].split("@")[0]),
                google_id=google_info["google_id"],
                picture=google_info.get("picture", ""),
            )
            db.add(user)
            await db.commit()
        token = create_token(user.id, user.email, user.name)
        refresh_tok = create_refresh_token()
        await store_refresh_token(user.id, refresh_tok, _refresh_expiry())
        _set_refresh_cookie(response, refresh_tok)
        logger.info("Google auth: %s", user.email)
        return AuthResponse(token=token, user={"id": user.id, "email": user.email, "name": user.name, "preferred_name": user.preferred_name or ""})


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    user_id = user.get("sub", "")
    preferred_name = ""
    try:
        factory = get_session_factory()
        async with factory() as db:
            db_user = await db.get(User, user_id)
            if db_user:
                preferred_name = db_user.preferred_name or ""
    except Exception as e:
        logger.warning("Failed to load preferred_name for %s: %s", user_id, e)
    return {
        "id": user_id,
        "email": user.get("email"),
        "name": user.get("name"),
        "preferred_name": preferred_name,
    }


@router.patch("/profile")
async def update_profile(
    request: ProfileUpdateRequest,
    user: dict = Depends(get_current_user),
):
    """Update the current user's personalization settings (preferred name)."""
    user_id = user.get("sub", "")
    factory = get_session_factory()
    async with factory() as db:
        db_user = await db.get(User, user_id)
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        db_user.preferred_name = request.preferred_name
        await db.commit()
        logger.info("Updated preferred_name for %s", db_user.email)
        return {
            "id": db_user.id,
            "email": db_user.email,
            "name": db_user.name,
            "preferred_name": db_user.preferred_name or "",
        }


@router.post("/refresh")
async def refresh_tokens(request: Request, response: Response):
    """
    Exchange a valid refresh token cookie for a new access token + rotated refresh token.
    The old refresh token is deleted atomically — reuse of a stolen token is detected here.
    """
    old_tok = request.cookies.get("refresh_token")
    if not old_tok:
        raise HTTPException(status_code=401, detail="No refresh token")
    result = await validate_and_rotate_refresh_token(old_tok)
    if result is None:
        _clear_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="Refresh token invalid or expired")
    user_id, new_tok = result
    factory = get_session_factory()
    async with factory() as db:
        user = await db.get(User, user_id)
        if not user:
            _clear_refresh_cookie(response)
            raise HTTPException(status_code=401, detail="User not found")
    _set_refresh_cookie(response, new_tok)
    token = create_token(user.id, user.email, user.name)
    logger.info("Token refreshed for user: %s", user.email)
    return AuthResponse(token=token, user={"id": user.id, "email": user.email, "name": user.name, "preferred_name": user.preferred_name or ""})


@router.post("/logout")
async def logout_user(request: Request, response: Response):
    """Revoke the refresh token and clear the cookie."""
    old_tok = request.cookies.get("refresh_token")
    if old_tok:
        await revoke_refresh_token(old_tok)
    _clear_refresh_cookie(response)
    return {"message": "Logged out"}


@router.get("/rate-limit")
async def get_rate_limit(user: dict = Depends(get_current_user)):
    """Return the current user's hourly query usage and remaining quota."""
    user_id = user.get("sub", "")
    settings = get_settings()
    limit = settings.rate_limit_per_hour
    redis = await get_redis()
    used = 0
    if redis:
        try:
            count = await redis.get(f"ratelimit:{user_id}")
            used = int(count) if count else 0
        except Exception:
            pass
    return {"used": used, "limit": limit, "remaining": max(0, limit - used)}
