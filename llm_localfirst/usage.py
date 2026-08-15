"""Token accounting and a fail-closed ceiling on cloud spend.

The privacy guarantee already answers *may this call leave the machine?*. This module
answers the other question a local-first setup has to answer: *how much has leaving the
machine already cost?* — and lets you cap it.

Three pieces, all pure except where noted:

- :class:`Usage` — normalised token counts, since providers disagree on the field names.
- :class:`Ledger` — an in-process running total, split by ``local`` / ``cloud``.
- :class:`Budget` — a ceiling checked *before* a cloud call is dispatched.

**Prices are configuration, not library knowledge.** This package does not ship a table
of per-model prices: they change, and a stale hard-coded number is worse than no number.
You supply them (``LF_PRICES``, or the ``prices`` argument), and a cost budget
refuses to be configured without them rather than silently never triggering.

**The ceiling is enforced between calls, not mid-call.** Token counts only exist once a
provider has answered, so a budget blocks the *next* cloud call after it is breached. It
bounds the overshoot to one call; it cannot bound a single call.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import BudgetExceeded
from .registry import ModelRef, Target

__all__ = ["Usage", "Price", "CallRecord", "Ledger", "Budget", "normalise_usage"]

#: Tokens are priced per million, the unit every provider publishes.
TOKENS_PER_PRICE_UNIT = 1_000_000


@dataclass(frozen=True)
class Usage:
    """Normalised token counts for one or more calls."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class Price:
    """Per-million-token prices for one model, in whatever currency you're counting."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_per_mtok + usage.output_tokens * self.output_per_mtok
        ) / TOKENS_PER_PRICE_UNIT


def _first_int(raw: dict, *keys: str) -> int:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def normalise_usage(raw: dict | None) -> Usage:
    """Read a provider's usage payload into a :class:`Usage`.

    OpenAI-compatible servers report ``prompt_tokens`` / ``completion_tokens``;
    Anthropic reports ``input_tokens`` / ``output_tokens``. A provider that reports
    nothing (many local servers) yields a zero :class:`Usage` — which is honest: we
    did not measure it, so we do not charge for it.

    Anthropic's cache-tier counts are added to the input side. They are billed at a
    different rate than ordinary input tokens, so a cost figure that includes heavy
    prompt caching is an approximation — the token counts stay exact.
    """
    if not isinstance(raw, dict):
        return Usage()
    return Usage(
        input_tokens=(
            _first_int(raw, "input_tokens", "prompt_tokens")
            + _first_int(raw, "cache_creation_input_tokens")
            + _first_int(raw, "cache_read_input_tokens")
        ),
        output_tokens=_first_int(raw, "output_tokens", "completion_tokens"),
    )


@dataclass(frozen=True)
class CallRecord:
    """One completed call, as accounted."""

    model: str
    target: Target
    usage: Usage
    #: ``None`` when the model has no configured price — distinct from a cost of 0.
    cost: float | None


class Ledger:
    """A running, in-process tally of token use and estimated spend.

    Scoped to one :class:`~llm_localfirst.router.Router` instance, in memory: it is a
    guard rail for a process, not billing. Restarting resets it. If you need spend
    enforced across processes, read :meth:`snapshot` into your own store.
    """

    def __init__(self, prices: dict[str, Price] | None = None) -> None:
        self.prices: dict[str, Price] = dict(prices or {})
        self._records: list[CallRecord] = []

    # -- writing --------------------------------------------------------------

    def record(self, model: ModelRef, usage: Usage) -> CallRecord:
        """Account one completed call and return the resulting record."""
        price = self.prices.get(model.name)
        record = CallRecord(
            model=model.name,
            target=model.target,
            usage=usage,
            cost=price.cost(usage) if price is not None else None,
        )
        self._records.append(record)
        return record

    def reset(self) -> None:
        """Drop every record. Prices are configuration and are kept."""
        self._records.clear()

    # -- reading --------------------------------------------------------------

    @property
    def records(self) -> list[CallRecord]:
        return list(self._records)

    def _select(self, target: Target | None) -> list[CallRecord]:
        if target is None:
            return self._records
        return [r for r in self._records if r.target == target]

    def calls(self, target: Target | None = None) -> int:
        return len(self._select(target))

    def tokens(self, target: Target | None = None) -> Usage:
        total = Usage()
        for record in self._select(target):
            total = total + record.usage
        return total

    def cost(self, target: Target | None = None) -> float:
        """Estimated spend over the priced calls.

        Calls against unpriced models contribute nothing — check
        :meth:`unpriced_calls` before reading this as a complete figure.
        """
        return sum(r.cost for r in self._select(target) if r.cost is not None)

    def unpriced_calls(self, target: Target | None = None) -> int:
        """How many accounted calls had no configured price."""
        return sum(1 for r in self._select(target) if r.cost is None)

    def by_model(self) -> dict[str, Usage]:
        totals: dict[str, Usage] = {}
        for record in self._records:
            totals[record.model] = totals.get(record.model, Usage()) + record.usage
        return totals

    def snapshot(self) -> dict:
        """A JSON-safe summary, for logging or handing to your own store."""
        return {
            "calls": {
                "total": self.calls(),
                "local": self.calls("local"),
                "cloud": self.calls("cloud"),
            },
            "tokens": {
                "total": self.tokens().as_dict(),
                "local": self.tokens("local").as_dict(),
                "cloud": self.tokens("cloud").as_dict(),
            },
            "cloud_cost": self.cost("cloud"),
            "unpriced_cloud_calls": self.unpriced_calls("cloud"),
            "by_model": {name: usage.as_dict() for name, usage in self.by_model().items()},
        }


@dataclass(frozen=True)
class Budget:
    """A ceiling on cloud usage, checked before each cloud dispatch.

    Every limit is optional; those left ``None`` are not enforced. Limits are
    inclusive ceilings — reaching exactly ``max_cloud_calls`` blocks the next call.

    ``max_cloud_calls`` and ``max_cloud_tokens`` are exact and need no configuration.
    ``max_cloud_cost`` needs a price for every allowlisted cloud model; a Router
    refuses to start without them, so a cost ceiling can never silently fail open.
    """

    max_cloud_calls: int | None = None
    max_cloud_tokens: int | None = None
    max_cloud_cost: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_cloud_calls", "max_cloud_tokens", "max_cloud_cost"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")

    @property
    def is_empty(self) -> bool:
        return (
            self.max_cloud_calls is None
            and self.max_cloud_tokens is None
            and self.max_cloud_cost is None
        )

    def exceeded(self, ledger: Ledger) -> str | None:
        """Return a human-readable reason if any limit is spent, else ``None``."""
        if self.max_cloud_calls is not None:
            calls = ledger.calls("cloud")
            if calls >= self.max_cloud_calls:
                return f"cloud call budget spent: {calls}/{self.max_cloud_calls} calls"

        if self.max_cloud_tokens is not None:
            tokens = ledger.tokens("cloud").total_tokens
            if tokens >= self.max_cloud_tokens:
                return f"cloud token budget spent: {tokens}/{self.max_cloud_tokens} tokens"

        if self.max_cloud_cost is not None:
            cost = ledger.cost("cloud")
            if cost >= self.max_cloud_cost:
                return f"cloud cost budget spent: {cost:.4f}/{self.max_cloud_cost:.4f}"

        return None

    def check(self, ledger: Ledger) -> None:
        """Raise :class:`BudgetExceeded` if any limit is spent."""
        reason = self.exceeded(ledger)
        if reason is not None:
            raise BudgetExceeded(reason)

    def require_prices_for(self, models: list[ModelRef], prices: dict[str, Price]) -> None:
        """Fail configuration when a cost ceiling can't actually be enforced.

        A cost budget over a model with no price would sit at 0.0 forever and never
        trigger — the exact failure mode a budget exists to prevent. Better to refuse
        at construction than to look protected in production.
        """
        if self.max_cloud_cost is None:
            return
        missing = sorted(m.name for m in models if m.name not in prices)
        if missing:
            raise ValueError(
                "max_cloud_cost needs a price for every allowlisted cloud model; "
                f"missing: {', '.join(missing)}. Set LF_PRICES, e.g. "
                "'{\"haiku\": [0.8, 4.0]}' (input and output price per million tokens), "
                "or use max_cloud_tokens / max_cloud_calls, which need no prices."
            )
