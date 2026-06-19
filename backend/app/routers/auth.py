"""
Auth router — register, login, Google OAuth, /me endpoint.
Also exports get_current_user and check_rate_limit for use by other routers.
"""

import uuid
import logging

from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy import select

from app.config import get_settings
from app.models.schemas import (
    RegisterRequest,
    LoginRequest,
    GoogleAuthRequest,
    AuthResponse,
)
from app.models.database import User, get_session_factory
from app.services.auth import (
    hash_password,
    verify_password,
    create_token,
    validate_token,
    verify_google_token,
)
from app.services.cache import get_redis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


async def get_current_user(request: Request) -> dict:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = auth_header[7:]
    payload = validate_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


async def check_rate_limit(user_id: str) -> None:
    redis = await get_redis()
    if redis is None:
        return
    settings = get_settings()
    key = f"ratelimit:{user_id}"
    try:
        count = await redis.get(key)
        if count and int(count) >= settings.rate_limit_per_hour:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {settings.rate_limit_per_hour} queries per hour.",
            )
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 3600)
        await pipe.execute()
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Rate limit check failed: %s", e)


@router.post("/register")
async def register(request: RegisterRequest):
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
        logger.info("New user registered: %s", user.email)
        return AuthResponse(token=token, user={"id": user.id, "email": user.email, "name": user.name})


@router.post("/login")
async def login(request: LoginRequest):
    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(select(User).where(User.email == request.email))
        user = result.scalar_one_or_none()
        if not user or not user.password_hash:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not verify_password(request.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_token(user.id, user.email, user.name)
        logger.info("User logged in: %s", user.email)
        return AuthResponse(token=token, user={"id": user.id, "email": user.email, "name": user.name})


@router.post("/google")
async def google_auth(request: GoogleAuthRequest):
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
        logger.info("Google auth: %s", user.email)
        return AuthResponse(token=token, user={"id": user.id, "email": user.email, "name": user.name})


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {"id": user.get("sub"), "email": user.get("email"), "name": user.get("name")}
