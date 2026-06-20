"""Backend protocol and the completion result type.

A *backend* is the thin async adapter that actually talks to one provider family
(local OpenAI-compatible servers, or Anthropic's Claude API). Backends are the only
place a provider SDK is imported, and they always import it **lazily** so that
``import llm_localfirst`` — and the whole :meth:`Router.decide` path — works with no
provider library installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..registry import ModelRef


@dataclass
class CompletionResult:
    """The outcome of a single completion call.

    Attributes:
        text: The generated text.
        model: The :class:`~llm_localfirst.registry.ModelRef` that produced it.
        usage: Provider-reported token usage, if the provider returned any.
    """

    text: str
    model: ModelRef
    usage: dict | None = None


@runtime_checkable
class Backend(Protocol):
    """Async adapter for one provider family.

    Implementations expose a ``provider`` string (matching
    :attr:`ModelRef.provider`) and a single async :meth:`complete` method.
    """

    provider: str

    async def complete(
        self,
        model: ModelRef,
        prompt: str,
        *,
        system: str | None = None,
        **opts: object,
    ) -> CompletionResult:
        """Run a completion for ``model`` and return its :class:`CompletionResult`."""
        ...
