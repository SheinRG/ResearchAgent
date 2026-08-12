"""
Anonymous visitor identity and the free-query allowance.

A logged-out visitor gets a small number of real research runs before the
signup wall. Two counters back that, and they do different jobs:

* **Per visitor (cookie).** The product limit. Small on purpose — see the note
  in ``config.py`` on why 3 converts better than 30.
* **Per IP.** A blunt backstop against clearing cookies for a fresh allowance.
  Deliberately more generous, because offices, campuses and mobile carriers put
  many people behind one address and a strict per-IP cap would lock them out
  collectively.

Neither is tamper-proof, and this module does not pretend otherwise:
``X-Forwarded-For`` is client-supplied and an incognito window is free. That is
tolerable because these are not the budget guard. The ceilings in
``services.budget`` are global, so they cannot be bypassed by minting new
identities — which is exactly why the money guard lives there and not here.

Both counters fail **closed**: when Redis is unavailable they fall back to
process-local counters rather than letting requests through uncounted, matching
how ``routers.auth`` handles the signed-in rate limit.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

from fastapi import HTTPException, Request, Response

from app.config import get_settings
from app.services.cache import get_redis

logger = logging.getLogger(__name__)

ANON_COOKIE = "goon_anon"

# Rolling 24h, matching how the allowance is described to the user ("3 free
# queries a day") rather than a calendar day, which would hand someone six in
# ten minutes across a midnight boundary.
_WINDOW_SECONDS = 86400

# Redis-down fallback, same shape and reasoning as auth._local_buckets: bounds
# abuse per-instance instead of removing the limit. Maps key -> (start, count).
_local_buckets: dict[str, tuple[float, int]] = {}


@dataclass(frozen=True)
class AnonQuota:
    """The free-query allowance for one visitor."""

    used: int
    limit: int
    anon_id: str

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def as_dict(self) -> dict:
        return {
            "used": self.used,
            "limit": self.limit,
            "remaining": self.remaining,
            "exhausted": self.exhausted,
        }


def new_anon_id() -> str:
    return uuid.uuid4().hex


def read_anon_id(request: Request) -> str:
    """The visitor's id from their cookie, or "" if they have none yet."""
    value = (request.cookies.get(ANON_COOKIE) or "").strip()
    # Only ever used as a Redis key suffix, but bound the length and alphabet
    # anyway — a cookie is attacker-controlled input.
    if len(value) == 32 and value.isalnum():
        return value
    return ""


def set_anon_cookie(response: Response, anon_id: str) -> None:
    """Attach the visitor cookie. httponly: nothing in the UI needs to read it."""
    settings = get_settings()
    secure = settings.environment == "production"
    response.set_cookie(
        key=ANON_COOKIE,
        value=anon_id,
        max_age=30 * 86400,
        httponly=True,
        secure=secure,
        # The demo frontend is a different origin from the API in production,
        # so the cookie has to survive a cross-site request to be worth setting.
        samesite="none" if secure else "lax",
        path="/",
    )


def client_ip(request: Request) -> str:
    """
    Best-effort client address.

    Render terminates TLS and forwards the original address in
    ``X-Forwarded-For``; the leftmost entry is the client. That entry is
    client-supplied and therefore spoofable — accepted deliberately, because
    this counter only blunts casual cookie-clearing and the global spend
    ceiling is what actually protects the budget.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first[:64]
    client = request.client
    return client.host if client else "unknown"


def _check_local(key: str, increment: bool) -> int:
    """Fixed-window counter in process memory (the Redis-down path)."""
    now = time.time()

    if len(_local_buckets) > 10_000:
        for k, (start, _) in list(_local_buckets.items()):
            if now - start >= _WINDOW_SECONDS:
                _local_buckets.pop(k, None)

    window_start, count = _local_buckets.get(key, (now, 0))
    if now - window_start >= _WINDOW_SECONDS:
        window_start, count = now, 0

    if increment:
        count += 1
        _local_buckets[key] = (window_start, count)
    return count


async def _counter(key: str, increment: bool) -> int:
    """Read (or bump) one window counter, falling back to process memory."""
    redis = await get_redis()
    if redis is None:
        return _check_local(key, increment)

    try:
        if not increment:
            value = await redis.get(key)
            return int(value or 0)
        # INCR then read, so concurrent requests cannot both see "one left".
        # The TTL is set only on the first hit of the window (nx) so repeated
        # requests cannot keep pushing the reset forward.
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, _WINDOW_SECONDS, nx=True)
        results = await pipe.execute()
        return int(results[0])
    except Exception as e:
        logger.warning("Anon quota (redis) failed, using local fallback: %s", e)
        return _check_local(key, increment)


def quota_exhausted(limit: int) -> HTTPException:
    """The signup wall, as a machine-readable error the frontend switches on."""
    return HTTPException(
        status_code=429,
        detail={
            "code": "demo_quota_exhausted",
            "limit": limit,
            "message": (
                f"You've used your {limit} free queries. "
                "Sign in to keep going — it's free."
            ),
        },
    )


async def peek_quota(request: Request, anon_id: str) -> AnonQuota:
    """
    How much allowance is left, without spending any.

    Read-only on purpose: the endpoint checks this *before* doing any work, so
    a visitor who is refused never has a query deducted for it.
    """
    settings = get_settings()
    limit = settings.anon_free_queries
    used = await _counter(f"anon:q:{anon_id}", increment=False) if anon_id else 0
    ip_used = await _counter(f"anon:ip:{client_ip(request)}", increment=False)

    # Report whichever cap is closer to biting, so the number the visitor sees
    # is the one that will actually stop them — a visitor on a busy office IP
    # should not be promised three queries and refused on their first.
    remaining = min(
        max(0, limit - used),
        max(0, settings.anon_free_queries_per_ip - ip_used),
    )
    return AnonQuota(used=limit - remaining, limit=limit, anon_id=anon_id)


async def consume_quota(request: Request, anon_id: str) -> AnonQuota:
    """
    Spend one free query, or raise the signup wall.

    Called only once the request is going to be served — including for an
    answer-cache hit, which costs nothing but is still a query the visitor got.
    The per-visitor allowance is a product decision about when to ask for a
    signup; what a run costs is the budget guard's business, not this one's.
    """
    settings = get_settings()
    limit = settings.anon_free_queries
    ip_limit = settings.anon_free_queries_per_ip

    used = await _counter(f"anon:q:{anon_id}", increment=True)
    ip_used = await _counter(f"anon:ip:{client_ip(request)}", increment=True)

    if used > limit or ip_used > ip_limit:
        raise quota_exhausted(limit)

    return AnonQuota(used=used, limit=limit, anon_id=anon_id)
