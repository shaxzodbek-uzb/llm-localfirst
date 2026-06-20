"""The Router — wires the pure :class:`~llm_localfirst.policy.Policy` to live I/O.

The Router is the runtime entry point for every privacy-bearing path (the CLI, the
MCP tools, the manager-worker delegation tool). It owns three collaborators:

- a :class:`~llm_localfirst.registry.Registry` (the model allowlist / SSRF guard),
- a :class:`~llm_localfirst.policy.Policy` (the pure decision brain),
- a :class:`~llm_localfirst.reachability.Reachability` probe (the only network I/O on
  the decision path),

plus a small map of provider :class:`~llm_localfirst.backends.base.Backend` adapters
used to actually run a completion.

:meth:`decide` probes local reachability (cached) and asks the Policy where a call
should go — no LLM is called. :meth:`acomplete` decides and then dispatches to the
matching backend. As defence in depth, :meth:`acomplete` re-asserts the headline
guarantee at the dispatch boundary: a ``sensitive`` call must resolve to a *local*
target before any backend is touched, so a future policy or registry bug can never
leak a sensitive prompt to the cloud.
"""

from __future__ import annotations

import asyncio

from .backends.anthropic import AnthropicBackend
from .backends.base import Backend, CompletionResult
from .backends.openai_compat import OpenAICompatBackend
from .config import Settings
from .errors import BackendError, PrivacyViolation
from .policy import Decision, Kind, Policy
from .reachability import Reachability
from .registry import Registry, default_registry

_SOURCE_SEPARATOR = "\n\n--- Source text ---\n"


class Router:
    """Routes completions local-first, with a fail-closed privacy guarantee."""

    def __init__(
        self,
        *,
        registry: Registry,
        policy: Policy,
        reachability: Reachability | None = None,
        backends: dict[str, Backend] | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Wire the Router from its collaborators.

        Args:
            registry: The model allowlist.
            policy: The pure routing policy (its ``local_name`` selects which model
                the reachability probe targets).
            reachability: Probe for the local endpoint. A fresh one is created if
                omitted.
            backends: Map of ``provider`` -> :class:`Backend`. Defaults to the two
                stock backends (``openai_compat`` and ``anthropic``); they import
                their provider SDK lazily, so constructing them needs nothing.
            settings: Optional settings kept for reference/introspection.
        """
        self.registry = registry
        self.policy = policy
        self.reachability = reachability or Reachability()
        self.backends: dict[str, Backend] = backends or {
            "openai_compat": OpenAICompatBackend(),
            "anthropic": AnthropicBackend(),
        }
        self.settings = settings

    @classmethod
    def from_env(cls) -> Router:
        """Build a Router from :class:`Settings` (env ``LF_*`` / ``.env``).

        Assembles the default registry, a policy honouring the configured fallback /
        reason models and the fail-closed flag, a reachability probe with the
        configured TTL, and the two stock backends. Backends are constructed
        tolerantly — they only import their provider SDK when a completion is run —
        so the :meth:`decide`-only path needs no provider library installed.
        """
        settings = Settings()
        registry = default_registry(settings)
        policy = Policy(
            registry,
            local_name="local",
            fallback_name=settings.fallback_model,
            reason_name=settings.reason_model,
            sensitive_fail_closed=settings.sensitive_fail_closed,
        )
        reachability = Reachability(ttl=settings.probe_ttl)
        return cls(
            registry=registry,
            policy=policy,
            reachability=reachability,
            settings=settings,
        )

    def decide(
        self,
        *,
        sensitive: bool = False,
        kind: Kind | str = Kind.AUTO,
        model: str | None = None,
    ) -> Decision:
        """Resolve where a call should run. Probes local reachability (cached) and
        delegates to :meth:`Policy.decide`; performs no LLM call.

        ``kind`` accepts a :class:`Kind` or its string value (coerced via
        ``Kind(value)``, which raises ``ValueError`` on an unknown kind).
        """
        kind = Kind(kind)
        local_ref = self.registry.get(self.policy.local_name)
        local_up = self.reachability.check(local_ref.base_url)
        return self.policy.decide(
            sensitive=sensitive, kind=kind, model=model, local_up=local_up
        )

    async def acomplete(
        self,
        prompt: str,
        *,
        source: str = "",
        system: str | None = None,
        sensitive: bool = False,
        kind: Kind | str = Kind.AUTO,
        model: str | None = None,
        **opts: object,
    ) -> CompletionResult:
        """Decide, then dispatch to the matching backend and return its result.

        When ``source`` is non-empty it is appended to ``prompt`` under a
        ``--- Source text ---`` separator. Extra keyword args are forwarded to the
        backend unchanged.

        Raises:
            PrivacyViolation: If a ``sensitive`` call somehow resolved to a cloud
                target (defence-in-depth — should be impossible given the Policy).
            BackendError: If no backend is registered for the chosen provider.
        """
        decision = self.decide(sensitive=sensitive, kind=kind, model=model)

        # Defence in depth: never dispatch a sensitive call to a cloud target, no
        # matter what the policy/registry returned. The Policy already guarantees
        # this; the assert here means a future bug fails closed instead of leaking.
        if sensitive and decision.target != "local":
            raise PrivacyViolation(
                f"refusing to dispatch a sensitive call to a {decision.target} "
                f"target ({decision.model.name}); sensitive data must stay local"
            )

        backend = self.backends.get(decision.model.provider)
        if backend is None:
            raise BackendError(
                f"no backend registered for provider {decision.model.provider!r} "
                f"(model {decision.model.name!r})"
            )

        full_prompt = prompt if not source else f"{prompt}{_SOURCE_SEPARATOR}{source}"
        return await backend.complete(
            decision.model, full_prompt, system=system, **opts
        )

    def complete(self, prompt: str, **kw: object) -> CompletionResult:
        """Synchronous convenience wrapper around :meth:`acomplete`.

        Runs the async path to completion with :func:`asyncio.run`, so it works
        from a plain script. Do not call it from inside a running event loop — use
        :meth:`acomplete` there instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acomplete(prompt, **kw))  # type: ignore[arg-type]
        raise RuntimeError(
            "Router.complete() cannot run inside an active event loop; "
            "await Router.acomplete() instead."
        )
