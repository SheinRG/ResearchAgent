"""
Unit tests for the global spend ceiling (app.services.budget).

The guard exists to stop anonymous demo traffic from spending a month of
free-tier search credits in an hour. Its two load-bearing properties are both
easy to break by accident, so both are pinned here:

  1. **It must never fail open.** An unreadable budget is not a spent budget of
     zero — every path that cannot establish spend must deny live research.
  2. **The demo has its own sub-pool**, so a spike of logged-out visitors
     throttles itself before it starves signed-in users.

No live Redis or Postgres — `evaluate` is pure so the policy can be tested
directly, and the I/O paths are faked.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services import budget
from app.services.budget import Spend, evaluate


# --- Fixtures ---------------------------------------------------------------


def _settings(**overrides):
    """A settings stand-in with the knobs `evaluate` reads."""
    base = dict(
        budget_guard_enabled=True,
        monthly_search_credits=600,
        daily_search_credits=120,
        anon_daily_search_credits=40,
        daily_cost_budget_usd=0.25,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _spend(**overrides):
    base = dict(
        credits_today=0,
        credits_month=0,
        anon_credits_today=0,
        cost_today_usd=0.0,
        known=True,
        source="db",
    )
    base.update(overrides)
    return Spend(**base)


# --- The fail-closed property ----------------------------------------------


def test_unknown_spend_denies_live_research():
    """
    The whole point of the guard. An unreadable budget must not read as an
    unspent one — a naive Redis-only counter would return "0 spent" here and
    authorise unlimited spend against a free tier.
    """
    status = evaluate(Spend(known=False), anonymous=True, settings=_settings())

    assert status.allowed is False
    assert status.reason == "unknown"
    assert status.message  # never a silent refusal


def test_unknown_spend_denies_signed_in_users_too():
    """Not knowing what has been spent is not a demo-only problem."""
    status = evaluate(Spend(known=False), anonymous=False, settings=_settings())

    assert status.allowed is False
    assert status.reason == "unknown"


@pytest.mark.asyncio
async def test_current_spend_falls_through_to_postgres_on_redis_miss():
    """A cold or evicted Redis cache must re-read the truth, not assume zero."""
    truth = _spend(credits_today=17, credits_month=42)

    with patch.object(budget, "_read_baseline", AsyncMock(return_value=None)), \
         patch.object(budget, "_spend_from_db", AsyncMock(return_value=truth)), \
         patch.object(budget, "_write_baseline", AsyncMock()), \
         patch.object(budget, "_read_reservations", AsyncMock(return_value=(0, 0))):
        result = await budget.current_spend()

    assert result.known is True
    assert result.credits_today == 17
    assert result.credits_month == 42


@pytest.mark.asyncio
async def test_current_spend_is_unknown_when_both_stores_are_down():
    """
    Redis down *and* Postgres down is the case that must not degrade to zero.
    `render.yaml` runs allkeys-lru, so an evicted counter and a dead cache look
    identical from here — the only safe answer is "I don't know".
    """
    with patch.object(budget, "_read_baseline", AsyncMock(return_value=None)), \
         patch.object(budget, "_spend_from_db", AsyncMock(return_value=None)):
        result = await budget.current_spend()

    assert result.known is False
    assert result.credits_today == 0  # zero, but explicitly not trusted
    assert evaluate(result, anonymous=True, settings=_settings()).allowed is False


@pytest.mark.asyncio
async def test_in_flight_reservations_count_against_the_ceiling():
    """
    Concurrent requests inside one 60s baseline window must not all read the
    same stale total and all be admitted — that is exactly how a launch spike
    drains a month before the first row is written.
    """
    with patch.object(
        budget, "_read_baseline", AsyncMock(return_value=_spend(credits_today=10))
    ), patch.object(budget, "_read_reservations", AsyncMock(return_value=(32, 24))):
        result = await budget.current_spend()

    assert result.credits_today == 42
    assert result.anon_credits_today == 24


# --- The ceilings themselves ------------------------------------------------


def test_monthly_ceiling_blocks_even_on_a_fresh_day():
    """The free tier is monthly, so a quiet day does not restore the month."""
    status = evaluate(
        _spend(credits_today=0, credits_month=600),
        anonymous=False,
        settings=_settings(),
    )

    assert status.allowed is False
    assert status.reason == "monthly"


def test_daily_ceiling_blocks_a_burst_within_budget_for_the_month():
    status = evaluate(
        _spend(credits_today=120, credits_month=200),
        anonymous=False,
        settings=_settings(),
    )

    assert status.allowed is False
    assert status.reason == "daily"


def test_cost_ceiling_catches_an_expensive_model_swap():
    """Credits usually bite first; this is the backstop when tokens get pricey."""
    status = evaluate(
        _spend(cost_today_usd=0.25), anonymous=False, settings=_settings()
    )

    assert status.allowed is False
    assert status.reason == "cost"


def test_headroom_allows_the_run():
    status = evaluate(
        _spend(credits_today=10, credits_month=100, anon_credits_today=5),
        anonymous=True,
        settings=_settings(),
    )

    assert status.allowed is True
    assert status.reason == ""


# --- The demo sub-pool ------------------------------------------------------


def test_demo_pool_exhaustion_blocks_visitors_but_not_signed_in_users():
    """
    The reason the sub-pool exists: a demo spike must throttle itself while the
    people who signed up keep working.
    """
    spent = _spend(credits_today=50, credits_month=200, anon_credits_today=40)
    settings = _settings()

    assert evaluate(spent, anonymous=True, settings=settings).allowed is False
    assert evaluate(spent, anonymous=True, settings=settings).reason == "anon_daily"
    assert evaluate(spent, anonymous=False, settings=settings).allowed is True


def test_global_ceiling_reported_ahead_of_the_demo_pool():
    """
    Ordering matters for honesty, not just correctness: when the whole month is
    gone, telling a visitor it is a demo limit implies signing in would help.
    """
    spent = _spend(credits_today=120, credits_month=600, anon_credits_today=40)

    status = evaluate(spent, anonymous=True, settings=_settings())

    assert status.reason == "monthly"


def test_guard_can_be_switched_off():
    """The escape hatch, for a paid tier or a local run."""
    status = evaluate(
        _spend(credits_today=9999, credits_month=9999),
        anonymous=True,
        settings=_settings(budget_guard_enabled=False),
    )

    assert status.allowed is True


# --- Reservation sizing -----------------------------------------------------


def test_reservation_is_the_worst_case_a_run_can_cost():
    """
    Admission control books the maximum, not an average: the actual figure is
    not known until the run finishes, and under-booking is what lets a burst
    through.
    """
    settings = SimpleNamespace(max_sub_queries=4, tavily_search_depth="basic")
    assert budget.max_credits_per_run(settings) == 4

    deep = SimpleNamespace(max_sub_queries=4, tavily_search_depth="advanced")
    assert budget.max_credits_per_run(deep) == 8


@pytest.mark.asyncio
async def test_reserve_without_redis_does_not_raise():
    """
    No reservation store is a degraded guard, not a broken one — the Postgres
    baseline still bounds spend, just with up to 60s of lag. It must not take
    the research request down with it.
    """
    with patch.object(budget, "get_redis", AsyncMock(return_value=None)):
        await budget.reserve(anonymous=True)  # must not raise


# --- Baseline serialization -------------------------------------------------


def test_spend_round_trips_through_the_redis_baseline():
    """`_write_baseline` stores `as_dict()`; `_read_baseline` must parse it."""
    original = _spend(
        credits_today=12, credits_month=34, anon_credits_today=5, cost_today_usd=0.01
    )

    parsed = json.loads(json.dumps(original.as_dict()))

    assert parsed["credits_today"] == 12
    assert parsed["credits_month"] == 34
    assert parsed["anon_credits_today"] == 5
    assert parsed["cost_today_usd"] == pytest.approx(0.01)
