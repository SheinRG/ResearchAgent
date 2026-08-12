"""
Global spend ceiling for live research.

This exists because anonymous demo traffic and a $0 budget pull against each
other. One research run costs up to ``max_sub_queries`` Tavily credits against a
1,000/month free tier, so a few hundred launch visitors can spend the month in
an hour. The ceiling is what makes serving logged-out visitors survivable: when
it is reached the app **degrades to cache-only with an honest banner**, which is
a far better failure than a dead demo during the one spike that matters.

Two properties do the real work, and both are easy to get wrong:

**It must not fail open.** The obvious design — ``INCRBYFLOAT`` a daily Redis
key — fails open on two independent paths. Redis is optional here and degrades
to ``None`` everywhere, so a ceiling reading a missing counter sees "$0 spent"
and authorises unlimited spend; and ``render.yaml`` sets ``allkeys-lru``, which
evicts *any* key including a no-TTL counter, so the budget silently resets
mid-day. That is a smoke alarm wired to the fuse it is watching. Ground truth is
therefore Postgres — ``research_queries`` already records ``search_credits`` and
``cost_usd`` per turn and cannot be evicted — with Redis only ever a 60s cache
in front of it. On a Redis miss we fall through to Postgres; if Postgres is also
unreachable we deny live research rather than guess.

**Credits, not dollars, are the binding constraint.** Groq's free tier
rate-limits and recovers within the hour. A spent Tavily credit is gone until
the month rolls over. The dollar ceiling is kept as a backstop against a model
swap making tokens expensive, but the credit ceilings are the ones that bite.

Anonymous traffic draws from a smaller sub-pool of the daily allowance, so a
demo spike throttles itself before it can starve signed-in users. Anonymous
spend is identified by ``user_id IS NULL``, which the schema already allows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import case, func, select

from app.config import get_settings
from app.models.database import ResearchQuery, get_session_factory
from app.services.cache import get_redis

logger = logging.getLogger(__name__)

# How long a Postgres sum is trusted before it is re-read. 60s is a deliberate
# trade: it keeps a launch spike off the database, and the in-flight reservation
# counter below covers the window so concurrent requests do not all read the
# same stale total and all get admitted.
_BASELINE_TTL_SECONDS = 60

# Reservations outlive their day only so a run started at 23:59 still resolves.
_RESERVATION_TTL_SECONDS = 2 * 86400

_BASELINE_KEY = "research:budget:baseline"


def _now() -> datetime:
    """Naive UTC, matching how ``created_at`` is stored."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _day_key(now: datetime) -> str:
    return now.strftime("%Y-%m-%d")


def _reservation_key(now: datetime, anonymous: bool) -> str:
    scope = "anon" if anonymous else "all"
    return f"research:budget:reserved:{scope}:{_day_key(now)}"


@dataclass(frozen=True)
class Spend:
    """What has actually been spent, from whichever source could answer."""

    credits_today: int = 0
    credits_month: int = 0
    anon_credits_today: int = 0
    cost_today_usd: float = 0.0
    # False when neither Postgres nor Redis could answer. Callers must treat an
    # unknown spend as a breach — see the module docstring.
    known: bool = False
    # "db", "redis", or "unknown" — surfaced on /api/health so a silently
    # degraded guard is visible rather than inferred.
    source: str = "unknown"

    def as_dict(self) -> dict:
        return {
            "credits_today": self.credits_today,
            "credits_month": self.credits_month,
            "anon_credits_today": self.anon_credits_today,
            "cost_today_usd": round(self.cost_today_usd, 6),
            "known": self.known,
            "source": self.source,
        }


@dataclass(frozen=True)
class BudgetStatus:
    """The admission decision for one live research run."""

    allowed: bool
    # "" when allowed. Otherwise a stable slug the frontend switches on:
    # "anon_daily", "daily", "monthly", "cost", "unknown".
    reason: str = ""
    message: str = ""
    spend: Spend = Spend()

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "message": self.message,
            "spend": self.spend.as_dict(),
        }


async def _spend_from_db(now: datetime) -> Spend | None:
    """Ground truth. Returns None only when Postgres cannot be reached."""
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = day_start.replace(day=1)

    try:
        factory = get_session_factory()
        async with factory() as db:
            # One round trip: today's totals and the month's credits together.
            # FILTER-style conditional aggregates keep this a single scan of the
            # created_at index rather than three separate queries.
            credits = func.coalesce(func.sum(ResearchQuery.search_credits), 0)
            result = await db.execute(
                select(
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    ResearchQuery.created_at >= day_start,
                                    ResearchQuery.search_credits,
                                ),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    (ResearchQuery.created_at >= day_start)
                                    & (ResearchQuery.user_id.is_(None)),
                                    ResearchQuery.search_credits,
                                ),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                    credits,
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    ResearchQuery.created_at >= day_start,
                                    ResearchQuery.cost_usd,
                                ),
                                else_=0.0,
                            )
                        ),
                        0.0,
                    ),
                ).where(ResearchQuery.created_at >= month_start)
            )
            today, anon_today, month, cost_today = result.one()
            return Spend(
                credits_today=int(today or 0),
                credits_month=int(month or 0),
                anon_credits_today=int(anon_today or 0),
                cost_today_usd=float(cost_today or 0.0),
                known=True,
                source="db",
            )
    except Exception as e:
        logger.error("Budget: could not read spend from Postgres: %s", e)
        return None


async def _read_baseline() -> Spend | None:
    """The last Postgres sum, if Redis still holds a fresh one."""
    redis = await get_redis()
    if redis is None:
        return None
    try:
        raw = await redis.get(_BASELINE_KEY)
        if not raw:
            return None
        data = json.loads(raw)
        return Spend(
            credits_today=int(data.get("credits_today", 0)),
            credits_month=int(data.get("credits_month", 0)),
            anon_credits_today=int(data.get("anon_credits_today", 0)),
            cost_today_usd=float(data.get("cost_today_usd", 0.0)),
            known=True,
            source="redis",
        )
    except Exception as e:
        logger.warning("Budget: baseline read failed: %s", e)
        return None


async def _write_baseline(spend: Spend, now: datetime) -> None:
    """Cache a fresh Postgres sum and clear the reservations it now includes."""
    redis = await get_redis()
    if redis is None:
        return
    try:
        await redis.set(
            _BASELINE_KEY,
            json.dumps(spend.as_dict()),
            ex=_BASELINE_TTL_SECONDS,
        )
        # Everything reserved before this read has now either landed in the sum
        # or was never spent, so the delta starts again from zero. A request
        # that commits between the read and this delete is counted twice for up
        # to 60s — an over-count, which is the safe direction to be wrong in.
        await redis.delete(
            _reservation_key(now, anonymous=False),
            _reservation_key(now, anonymous=True),
        )
    except Exception as e:
        logger.warning("Budget: baseline write failed: %s", e)


async def _read_reservations(now: datetime) -> tuple[int, int]:
    """Credits admitted since the last baseline read, as (all, anonymous)."""
    redis = await get_redis()
    if redis is None:
        return (0, 0)
    try:
        values = await redis.mget(
            _reservation_key(now, anonymous=False),
            _reservation_key(now, anonymous=True),
        )
        return (int(values[0] or 0), int(values[1] or 0))
    except Exception as e:
        logger.warning("Budget: reservation read failed: %s", e)
        return (0, 0)


async def current_spend() -> Spend:
    """
    Spend so far today and this month, including runs still in flight.

    Postgres is the source of truth; Redis holds it for 60s. A Redis miss falls
    through to Postgres, **never** to zero. When neither answers, the returned
    Spend has ``known=False`` and callers must deny live research.
    """
    now = _now()

    baseline = await _read_baseline()
    if baseline is None:
        fresh = await _spend_from_db(now)
        if fresh is None:
            # Both stores are unreachable. Not "nothing has been spent" — we do
            # not know, and the guard's whole job is to refuse to guess here.
            return Spend(known=False, source="unknown")
        await _write_baseline(fresh, now)
        baseline = fresh

    reserved_all, reserved_anon = await _read_reservations(now)
    return Spend(
        credits_today=baseline.credits_today + reserved_all,
        credits_month=baseline.credits_month + reserved_all,
        anon_credits_today=baseline.anon_credits_today + reserved_anon,
        cost_today_usd=baseline.cost_today_usd,
        known=True,
        source=baseline.source,
    )


def evaluate(spend: Spend, anonymous: bool, settings=None) -> BudgetStatus:
    """
    Decide whether a live run may proceed, given a spend snapshot.

    Pure and synchronous so the policy can be tested without Redis, Postgres, or
    a clock. ``current_spend`` supplies the numbers; this decides on them.
    """
    settings = settings or get_settings()

    if not settings.budget_guard_enabled:
        return BudgetStatus(allowed=True, spend=spend)

    if not spend.known:
        return BudgetStatus(
            allowed=False,
            reason="unknown",
            message=(
                "Live research is paused: the spend guard can't reach its "
                "budget records, and it won't spend against an unknown balance. "
                "Cached answers still work."
            ),
            spend=spend,
        )

    # Monthly first — it is the hard truth about the free tier, and reporting it
    # as a daily cap would be misleading when the day's allowance is untouched.
    if spend.credits_month >= settings.monthly_search_credits:
        return BudgetStatus(
            allowed=False,
            reason="monthly",
            message=(
                "This month's free search allowance is spent. Live research "
                "resumes when it resets; cached answers still work."
            ),
            spend=spend,
        )

    if spend.credits_today >= settings.daily_search_credits:
        return BudgetStatus(
            allowed=False,
            reason="daily",
            message=(
                "Today's free search allowance is spent. Live research resumes "
                "tomorrow; cached answers still work."
            ),
            spend=spend,
        )

    if spend.cost_today_usd >= settings.daily_cost_budget_usd:
        return BudgetStatus(
            allowed=False,
            reason="cost",
            message=(
                "Today's model budget is spent. Live research resumes "
                "tomorrow; cached answers still work."
            ),
            spend=spend,
        )

    # The demo sub-pool, checked last so a signed-in user never sees a reason
    # that does not apply to them. This is what stops a launch spike from
    # eating the allowance that signed-in users are relying on.
    if anonymous and spend.anon_credits_today >= settings.anon_daily_search_credits:
        return BudgetStatus(
            allowed=False,
            reason="anon_daily",
            message=(
                "The free demo has used today's live-research allowance. "
                "Answers already in the cache still work — or sign in to use "
                "the signed-in allowance."
            ),
            spend=spend,
        )

    return BudgetStatus(allowed=True, spend=spend)


async def check_budget(anonymous: bool) -> BudgetStatus:
    """Read spend and decide. The one call the request path needs."""
    settings = get_settings()
    if not settings.budget_guard_enabled:
        return BudgetStatus(allowed=True, spend=Spend(known=True, source="disabled"))
    return evaluate(await current_spend(), anonymous, settings)


async def reserve(anonymous: bool, credits: int | None = None) -> None:
    """
    Book the worst-case cost of an admitted run against the in-flight counters.

    Called at admission, before the run knows what it will actually spend, so
    the reservation is the *maximum* a run can cost. Under-reserving would let
    every request inside one 60s baseline window read the same stale total and
    all get admitted — which is precisely how a launch spike drains a month.
    The real figures land in Postgres when the turn is saved, and the next
    baseline read supersedes this.
    """
    settings = get_settings()
    if not settings.budget_guard_enabled:
        return

    if credits is None:
        credits = max_credits_per_run(settings)

    redis = await get_redis()
    if redis is None:
        # No reservation store. The Postgres baseline still bounds spend, just
        # with up to 60s of lag — logged so a quiet degradation is visible.
        logger.warning("Budget: no Redis, in-flight reservations are not tracked")
        return

    now = _now()
    try:
        pipe = redis.pipeline()
        pipe.incrby(_reservation_key(now, anonymous=False), credits)
        pipe.expire(_reservation_key(now, anonymous=False), _RESERVATION_TTL_SECONDS)
        if anonymous:
            pipe.incrby(_reservation_key(now, anonymous=True), credits)
            pipe.expire(_reservation_key(now, anonymous=True), _RESERVATION_TTL_SECONDS)
        await pipe.execute()
    except Exception as e:
        logger.warning("Budget: reservation failed (spend still bounded by db): %s", e)


def max_credits_per_run(settings=None) -> int:
    """Worst-case search credits one research turn can bill."""
    from app.services.usage import search_credits_for

    settings = settings or get_settings()
    return settings.max_sub_queries * search_credits_for(settings.tavily_search_depth)
