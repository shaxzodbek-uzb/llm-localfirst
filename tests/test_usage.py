"""Tests for usage accounting and the fail-closed cloud budget.

Offline throughout: a fake backend returns canned provider-shaped usage payloads, and
a stub reachability probe stands in for the network.
"""

from __future__ import annotations

import pytest

from llm_localfirst import (
    Budget,
    BudgetExceeded,
    CompletionResult,
    Ledger,
    ModelRef,
    Price,
    Router,
    Usage,
)
from llm_localfirst.usage import normalise_usage


class StubReachability:
    def __init__(self, up: bool = True) -> None:
        self.up = up

    def check(self, base_url: str | None, *, timeout: float = 2.0) -> bool:
        return self.up


class UsageBackend:
    """Fake backend returning a fixed, provider-shaped usage payload per call."""

    def __init__(self, provider: str, usage: dict | None) -> None:
        self.provider = provider
        self.usage = usage
        self.calls = 0

    async def complete(self, model, prompt, *, system=None, **opts) -> CompletionResult:
        self.calls += 1
        return CompletionResult(text="ok", model=model, usage=self.usage)


def _router(registry, policy, *, usage=None, ledger=None, budget=None, up=True) -> Router:
    payload = {"input_tokens": 100, "output_tokens": 50} if usage is None else usage
    return Router(
        registry=registry,
        policy=policy,
        reachability=StubReachability(up),
        backends={
            "openai_compat": UsageBackend("openai_compat", payload),
            "anthropic": UsageBackend("anthropic", payload),
        },
        ledger=ledger,
        budget=budget,
    )


# -- normalising provider payloads -------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Anthropic
        ({"input_tokens": 10, "output_tokens": 3}, Usage(10, 3)),
        # OpenAI-compatible
        ({"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}, Usage(10, 3)),
        # Anthropic prompt caching — cache tiers count on the input side
        (
            {
                "input_tokens": 10,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 2,
                "output_tokens": 3,
            },
            Usage(17, 3),
        ),
        # Local servers frequently report nothing at all
        (None, Usage()),
        ({}, Usage()),
        ("not a dict", Usage()),
        # Junk values must not crash the accounting path
        ({"input_tokens": None, "output_tokens": "many"}, Usage()),
        ({"input_tokens": True, "output_tokens": False}, Usage()),
    ],
)
def test_normalise_usage(raw, expected) -> None:
    assert normalise_usage(raw) == expected


def test_usage_adds_and_totals() -> None:
    total = Usage(10, 2) + Usage(5, 3)
    assert total == Usage(15, 5)
    assert total.total_tokens == 20


# -- pricing -----------------------------------------------------------------


def test_price_is_per_million_tokens() -> None:
    price = Price(input_per_mtok=3.0, output_per_mtok=15.0)
    # 1M input + 1M output
    assert price.cost(Usage(1_000_000, 1_000_000)) == pytest.approx(18.0)
    assert price.cost(Usage(1_000, 500)) == pytest.approx(0.003 + 0.0075)


def test_unpriced_model_records_none_not_zero() -> None:
    ledger = Ledger()
    model = ModelRef(name="haiku", target="cloud", provider="anthropic", model_id="x")
    record = ledger.record(model, Usage(1000, 1000))
    assert record.cost is None
    assert ledger.cost("cloud") == 0.0
    assert ledger.unpriced_calls("cloud") == 1


# -- the ledger --------------------------------------------------------------


@pytest.mark.asyncio
async def test_completions_are_accounted_by_target(registry, policy) -> None:
    router = _router(registry, policy)
    await router.acomplete("bulk work")  # local is up -> local
    await router.acomplete("reason", kind="reason")  # -> cloud

    assert router.ledger.calls() == 2
    assert router.ledger.calls("local") == 1
    assert router.ledger.calls("cloud") == 1
    assert router.ledger.tokens("cloud") == Usage(100, 50)
    assert router.ledger.tokens().total_tokens == 300


@pytest.mark.asyncio
async def test_a_provider_reporting_no_usage_is_counted_as_zero(registry, policy) -> None:
    """Honest accounting: we didn't measure it, so we don't charge for it."""
    router = _router(registry, policy, usage={})
    await router.acomplete("hello")
    assert router.ledger.calls() == 1
    assert router.ledger.tokens() == Usage()


@pytest.mark.asyncio
async def test_ledger_breaks_down_by_model(registry, policy) -> None:
    router = _router(registry, policy)
    await router.acomplete("a")
    await router.acomplete("b", kind="reason")
    assert set(router.ledger.by_model()) == {"local", "haiku"}


@pytest.mark.asyncio
async def test_snapshot_is_json_safe(registry, policy) -> None:
    import json

    router = _router(registry, policy)
    await router.acomplete("a")
    json.dumps(router.ledger.snapshot())  # must not raise


def test_reset_clears_records_but_keeps_prices() -> None:
    ledger = Ledger({"haiku": Price(1.0, 1.0)})
    model = ModelRef(name="haiku", target="cloud", provider="anthropic", model_id="x")
    ledger.record(model, Usage(10, 10))
    ledger.reset()
    assert ledger.calls() == 0
    assert "haiku" in ledger.prices


# -- the budget --------------------------------------------------------------


def test_budget_rejects_non_positive_limits() -> None:
    for kwargs in ({"max_cloud_calls": 0}, {"max_cloud_tokens": -1}, {"max_cloud_cost": 0.0}):
        with pytest.raises(ValueError, match="must be positive"):
            Budget(**kwargs)


def test_empty_budget_is_detectable() -> None:
    assert Budget().is_empty is True
    assert Budget(max_cloud_calls=1).is_empty is False


@pytest.mark.asyncio
async def test_call_budget_blocks_the_next_cloud_call(registry, policy) -> None:
    router = _router(registry, policy, budget=Budget(max_cloud_calls=2))
    await router.acomplete("a", kind="reason")
    await router.acomplete("b", kind="reason")
    with pytest.raises(BudgetExceeded, match="2/2 calls"):
        await router.acomplete("c", kind="reason")


@pytest.mark.asyncio
async def test_a_spent_budget_never_blocks_local(registry, policy) -> None:
    """Gating local calls would defeat the point of running local."""
    router = _router(registry, policy, budget=Budget(max_cloud_calls=1))
    await router.acomplete("a", kind="reason")  # spends the cloud budget
    for _ in range(5):
        await router.acomplete("bulk")  # local, still fine
    assert router.ledger.calls("local") == 5


@pytest.mark.asyncio
async def test_token_budget_blocks_once_breached(registry, policy) -> None:
    router = _router(registry, policy, budget=Budget(max_cloud_tokens=200))
    await router.acomplete("a", kind="reason")  # 150 tokens — under
    await router.acomplete("b", kind="reason")  # 300 total — over
    with pytest.raises(BudgetExceeded, match="300/200 tokens"):
        await router.acomplete("c", kind="reason")


@pytest.mark.asyncio
async def test_cost_budget_blocks_once_breached(registry, policy) -> None:
    ledger = Ledger({name: Price(10.0, 10.0) for name in ("haiku", "sonnet", "opus", "gpt")})
    router = _router(registry, policy, ledger=ledger, budget=Budget(max_cloud_cost=0.002))
    # 150 tokens at $10/Mtok = $0.0015 per call.
    await router.acomplete("a", kind="reason")
    await router.acomplete("b", kind="reason")  # $0.003 total — over
    with pytest.raises(BudgetExceeded, match="cloud cost budget spent"):
        await router.acomplete("c", kind="reason")


@pytest.mark.asyncio
async def test_the_overshoot_is_bounded_to_one_call(registry, policy) -> None:
    """Token counts only exist after the call, so the ceiling is enforced between calls."""
    router = _router(registry, policy, budget=Budget(max_cloud_tokens=10))
    await router.acomplete("a", kind="reason")  # 150 tokens, well past 10
    assert router.ledger.tokens("cloud").total_tokens == 150
    with pytest.raises(BudgetExceeded):
        await router.acomplete("b", kind="reason")


def test_cost_budget_refuses_to_start_without_prices(registry, policy) -> None:
    """A cost ceiling over unpriced models would sit at 0.0 and never fire."""
    with pytest.raises(ValueError, match="needs a price for every allowlisted cloud model"):
        Router(
            registry=registry,
            policy=policy,
            reachability=StubReachability(True),
            budget=Budget(max_cloud_cost=1.0),
        )


def test_cost_budget_starts_when_every_cloud_model_is_priced(registry, policy) -> None:
    ledger = Ledger({name: Price(1.0, 1.0) for name in ("haiku", "sonnet", "opus", "gpt")})
    router = Router(
        registry=registry,
        policy=policy,
        reachability=StubReachability(True),
        ledger=ledger,
        budget=Budget(max_cloud_cost=1.0),
    )
    assert router.budget is not None


def test_token_and_call_budgets_need_no_prices(registry, policy) -> None:
    """The two exact limits work with zero configuration."""
    router = Router(
        registry=registry,
        policy=policy,
        reachability=StubReachability(True),
        budget=Budget(max_cloud_calls=5, max_cloud_tokens=1000),
    )
    assert router.ledger.prices == {}


@pytest.mark.asyncio
async def test_a_blocked_call_is_not_accounted(registry, policy) -> None:
    """The budget check runs before dispatch, so nothing is recorded for a refusal."""
    router = _router(registry, policy, budget=Budget(max_cloud_calls=1))
    await router.acomplete("a", kind="reason")
    with pytest.raises(BudgetExceeded):
        await router.acomplete("b", kind="reason")
    assert router.ledger.calls("cloud") == 1


@pytest.mark.asyncio
async def test_no_budget_means_accounting_without_a_ceiling(registry, policy) -> None:
    router = _router(registry, policy)
    for _ in range(10):
        await router.acomplete("x", kind="reason")
    assert router.ledger.calls("cloud") == 10


# -- from_env wiring ---------------------------------------------------------


def test_from_env_leaves_the_budget_off_by_default(monkeypatch) -> None:
    for var in ("LF_MAX_CLOUD_CALLS", "LF_MAX_CLOUD_TOKENS", "LF_MAX_CLOUD_COST", "LF_PRICES"):
        monkeypatch.delenv(var, raising=False)
    router = Router.from_env()
    assert router.budget is None
    assert router.ledger.prices == {}


def test_from_env_reads_prices_and_limits(monkeypatch) -> None:
    monkeypatch.setenv("LF_PRICES", '{"haiku": [0.8, 4.0], "sonnet": [3.0, 15.0]}')
    monkeypatch.setenv("LF_MAX_CLOUD_TOKENS", "50000")
    monkeypatch.delenv("LF_MAX_CLOUD_COST", raising=False)
    monkeypatch.delenv("LF_MAX_CLOUD_CALLS", raising=False)

    router = Router.from_env()
    assert router.budget is not None
    assert router.budget.max_cloud_tokens == 50_000
    assert router.ledger.prices["haiku"] == Price(0.8, 4.0)
