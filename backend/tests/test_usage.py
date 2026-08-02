"""
Unit tests for per-request token/cost accounting.

The accumulator is what turns the (previously log-only) TokenUsage reports into
a number the user sees, so the tests that matter are: does an unknown model fail
soft rather than inventing a price, and does the ContextVar scope actually
survive the hop into asyncio.create_task — which is how the real request works,
since the graph runs in a task spawned by the endpoint.
"""

import asyncio

import pytest

from app.services.usage import (
    MODEL_PRICES,
    TokenUsage,
    UsageAccumulator,
    cost_of,
    current_usage,
    empty_usage,
    price_for,
    record_usage,
    usage_scope,
)

SYNTH_MODEL = "llama-3.3-70b-versatile"
FAST_MODEL = "llama-3.1-8b-instant"


def usage(stage="synthesis", model=SYNTH_MODEL, prompt=1000, completion=500):
    return TokenUsage(
        stage=stage,
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


# ===========================================================================
# Pricing
# ===========================================================================

class TestPricing:

    def test_known_model_has_a_price(self):
        assert price_for(SYNTH_MODEL) == MODEL_PRICES[SYNTH_MODEL]

    def test_unknown_model_costs_zero_rather_than_guessing(self):
        assert price_for("some-model-we-never-heard-of") == (0.0, 0.0)
        assert cost_of("some-model-we-never-heard-of", 1_000_000, 1_000_000) == 0.0

    def test_cost_is_per_million_tokens(self):
        prompt_price, completion_price = MODEL_PRICES[SYNTH_MODEL]
        assert cost_of(SYNTH_MODEL, 1_000_000, 0) == pytest.approx(prompt_price)
        assert cost_of(SYNTH_MODEL, 0, 1_000_000) == pytest.approx(completion_price)

    def test_completion_tokens_cost_more_than_prompt_tokens(self):
        """Sanity check on the table itself — output is never cheaper than input."""
        for model, (prompt_price, completion_price) in MODEL_PRICES.items():
            assert completion_price >= prompt_price, model

    def test_zero_tokens_cost_nothing(self):
        assert cost_of(SYNTH_MODEL, 0, 0) == 0.0


# ===========================================================================
# UsageAccumulator
# ===========================================================================

class TestAccumulator:

    def test_starts_empty(self):
        acc = UsageAccumulator()
        assert acc.calls == 0
        assert acc.total_tokens == 0
        assert acc.cost_usd == 0.0
        assert acc.by_stage == {}

    def test_sums_across_calls(self):
        acc = UsageAccumulator()
        acc.add(usage(prompt=100, completion=50))
        acc.add(usage(prompt=200, completion=25))

        assert acc.calls == 2
        assert acc.prompt_tokens == 300
        assert acc.completion_tokens == 75
        assert acc.total_tokens == 375

    def test_attributes_cost_per_stage(self):
        acc = UsageAccumulator()
        acc.add(usage(stage="triage", model=FAST_MODEL, prompt=800, completion=100))
        acc.add(usage(stage="synthesis", model=SYNTH_MODEL, prompt=4000, completion=700))
        acc.add(usage(stage="synthesis", model=SYNTH_MODEL, prompt=100, completion=10))

        assert set(acc.by_stage) == {"triage", "synthesis"}
        assert acc.by_stage["synthesis"].calls == 2
        assert acc.by_stage["triage"].calls == 1
        # Stage costs must add up to the total, or the breakdown is a lie.
        assert sum(s.cost_usd for s in acc.by_stage.values()) == pytest.approx(acc.cost_usd)

    def test_synthesis_dominates_cost(self):
        """The 70B answer call should cost far more than the 8B triage call."""
        acc = UsageAccumulator()
        acc.add(usage(stage="triage", model=FAST_MODEL, prompt=800, completion=100))
        acc.add(usage(stage="synthesis", model=SYNTH_MODEL, prompt=4000, completion=700))

        assert acc.by_stage["synthesis"].cost_usd > acc.by_stage["triage"].cost_usd * 10

    def test_missing_stage_label_is_bucketed_not_dropped(self):
        acc = UsageAccumulator()
        acc.add(usage(stage=""))
        assert acc.by_stage["unknown"].calls == 1

    def test_as_dict_shape(self):
        acc = UsageAccumulator()
        acc.add(usage(stage="synthesis"))
        payload = acc.as_dict()

        assert payload["calls"] == 1
        assert payload["total_tokens"] == 1500
        assert payload["cost_usd"] > 0
        assert payload["by_stage"]["synthesis"]["calls"] == 1

    def test_as_dict_keeps_sub_cent_precision(self):
        """A cheap call must not round to $0.00 — that would empty the cost line."""
        acc = UsageAccumulator()
        acc.add(usage(stage="triage", model=FAST_MODEL, prompt=500, completion=50))
        assert acc.as_dict()["cost_usd"] > 0

    def test_empty_usage_is_all_zeros(self):
        payload = empty_usage()
        assert payload["calls"] == 0
        assert payload["total_tokens"] == 0
        assert payload["cost_usd"] == 0.0


# ===========================================================================
# Scoping
# ===========================================================================

class TestUsageScope:

    def test_record_outside_a_scope_is_a_noop(self):
        assert current_usage() is None
        record_usage(usage())  # must not raise

    def test_scope_collects_records(self):
        with usage_scope() as acc:
            record_usage(usage(prompt=10, completion=5))
            assert acc.total_tokens == 15

    def test_scope_is_cleared_on_exit(self):
        with usage_scope():
            assert current_usage() is not None
        assert current_usage() is None

    def test_nested_scopes_do_not_leak_into_each_other(self):
        with usage_scope() as outer:
            record_usage(usage(prompt=10, completion=0))
            with usage_scope() as inner:
                record_usage(usage(prompt=999, completion=0))
            assert inner.prompt_tokens == 999
            assert outer.prompt_tokens == 10

    @pytest.mark.asyncio
    async def test_accumulator_survives_into_a_spawned_task(self):
        """
        This is the mechanism the endpoint depends on: it opens the scope, then
        spawns the graph in a task. asyncio copies the context into the task, so
        usage recorded inside the task must land in the caller's accumulator.
        """
        async def run_graph():
            record_usage(usage(stage="triage", prompt=100, completion=20))
            await asyncio.sleep(0)
            record_usage(usage(stage="synthesis", prompt=4000, completion=700))

        with usage_scope() as acc:
            task = asyncio.create_task(run_graph())
            await task

        assert acc.calls == 2
        assert acc.total_tokens == 4820
        assert set(acc.by_stage) == {"triage", "synthesis"}

    @pytest.mark.asyncio
    async def test_concurrent_requests_keep_separate_totals(self):
        """Two in-flight requests must not bill each other's tokens."""
        async def request(prompt_tokens):
            with usage_scope() as acc:
                task = asyncio.create_task(_record_after_yield(prompt_tokens))
                await task
                return acc.prompt_tokens

        async def _record_after_yield(prompt_tokens):
            await asyncio.sleep(0)
            record_usage(usage(prompt=prompt_tokens, completion=0))

        first, second = await asyncio.gather(request(100), request(9000))
        assert first == 100
        assert second == 9000
