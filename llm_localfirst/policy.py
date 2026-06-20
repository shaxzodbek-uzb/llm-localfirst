"""Routing policy — the pure decision brain of llm-localfirst.

``Policy.decide`` is a side-effect-free function: it takes the caller's intent
(``sensitive``, ``kind``, an optional explicit ``model``) plus a single observed
fact (``local_up``) and returns a :class:`Decision`. All I/O — probing whether the
local model is reachable — lives in the Router, so this module is trivially
testable offline.

The headline guarantee lives here: a ``sensitive=True`` call is **pinned to the
local model** and is *never* allowed to resolve to a cloud target. When the local
model is down, a sensitive call **fails closed** (raises :class:`LocalUnavailable`)
instead of leaking the prompt to a cloud provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import LocalUnavailable, PrivacyViolation
from .registry import ModelRef, Registry, Target


class Kind(str, Enum):
    """The nature of the work, which biases routing.

    - ``BULK``: token-heavy, low-risk text labor (summarize / draft / translate /
      reformat / extract / classify). Prefer the local model.
    - ``REASON``: planning, decisions, hard reasoning. Prefer the cloud model.
    - ``AUTO``: the default. Prefer local (for cost and privacy); use the cloud
      only when the local model is unreachable.
    """

    BULK = "bulk"
    REASON = "reason"
    AUTO = "auto"


@dataclass(frozen=True)
class Decision:
    """The resolved routing decision returned by :meth:`Policy.decide`.

    Attributes:
        model: The concrete :class:`~llm_localfirst.registry.ModelRef` to call.
        target: ``"local"`` or ``"cloud"`` — where the call will go.
        reason: A short human-readable explanation, e.g. ``"sensitive -> pinned local"``.
        fell_back: ``True`` iff we left the *preferred* target because the local
            model was unreachable. Always ``False`` for sensitive calls (which fail
            closed rather than fall back).
    """

    model: ModelRef
    target: Target
    reason: str
    fell_back: bool


class Policy:
    """Pure routing policy over an allowlist :class:`~llm_localfirst.registry.Registry`.

    The policy never performs I/O. The Router supplies ``local_up`` (from a cached
    reachability probe) so that :meth:`decide` stays a pure function of its inputs.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        local_name: str = "local",
        fallback_name: str = "haiku",
        reason_name: str = "haiku",
        sensitive_fail_closed: bool = True,
    ) -> None:
        """Configure the policy.

        Args:
            registry: The allowlist of resolvable models.
            local_name: Allowlist key of the local model (the privacy-safe target).
            fallback_name: Allowlist key used when a non-sensitive bulk/auto call
                must leave local because the local model is down.
            reason_name: Allowlist key used for ``Kind.REASON`` work (cloud).
            sensitive_fail_closed: When ``True`` (default), a sensitive call whose
                local model is unreachable raises :class:`LocalUnavailable` instead
                of ever touching the cloud. This is the product's headline guarantee.
        """
        self.registry = registry
        self.local_name = local_name
        self.fallback_name = fallback_name
        self.reason_name = reason_name
        self.sensitive_fail_closed = sensitive_fail_closed

    def decide(
        self,
        *,
        sensitive: bool,
        kind: Kind,
        model: str | None,
        local_up: bool,
    ) -> Decision:
        """Resolve a routing :class:`Decision`. PURE — performs no I/O.

        Rules, applied in order:

        1. **Explicit override.** If ``model`` is given it must be allowlisted
           (else :class:`~llm_localfirst.errors.ModelNotAllowed` via the registry).
           If the call is ``sensitive`` and the chosen model targets the cloud,
           raise :class:`PrivacyViolation`. Otherwise honour it (``fell_back=False``).
        2. **Sensitive (fail-closed).** ``sensitive=True`` pins to the local model.
           If the local model is not up and ``sensitive_fail_closed`` is set,
           raise :class:`LocalUnavailable`. A sensitive call NEVER silently falls
           back to the cloud.
        3. **Reason.** ``kind == Kind.REASON`` uses ``reason_name`` (cloud).
        4. **Bulk / Auto.** Prefer local: if ``local_up`` use the local model;
           otherwise fall back to ``fallback_name`` (cloud) with ``fell_back=True``.

        Args:
            sensitive: Whether the prompt carries data that must stay local.
            kind: The nature of the work (see :class:`Kind`).
            model: Optional explicit allowlist key overriding kind-based routing.
            local_up: Whether the local model is currently reachable (supplied by
                the Router's cached reachability probe).

        Returns:
            The resolved :class:`Decision`.

        Raises:
            ModelNotAllowed: If ``model`` is given but not on the allowlist.
            PrivacyViolation: If a sensitive call explicitly overrides to a cloud model.
            LocalUnavailable: If a sensitive call needs the local model but it is
                down and ``sensitive_fail_closed`` is set.
        """
        # 1. Explicit model override (allowlist-checked).
        if model is not None:
            ref = self.registry.get(model)  # raises ModelNotAllowed if absent
            if sensitive and ref.target == "cloud":
                raise PrivacyViolation(
                    f"sensitive call refuses cloud model {model!r} "
                    f"(target=cloud); sensitive data must stay local"
                )
            return Decision(
                model=ref,
                target=ref.target,
                reason=f"explicit override -> {model}",
                fell_back=False,
            )

        # 2. Sensitive: pin to local, fail closed if local is down.
        if sensitive:
            if not local_up and self.sensitive_fail_closed:
                raise LocalUnavailable(
                    "sensitive call requires the local model, but it is unreachable; "
                    "failing closed (will not leak the prompt to the cloud)"
                )
            ref = self.registry.get(self.local_name)
            # Structural guarantee: the "local" allowlist key MUST resolve to a local
            # target. If it is mis-wired to a cloud model, refuse rather than leak a
            # sensitive prompt — the guarantee must not depend on correct config.
            if ref.target != "local":
                raise PrivacyViolation(
                    f"local model {self.local_name!r} resolves to a {ref.target} "
                    f"target; a sensitive call cannot proceed without a local model"
                )
            return Decision(
                model=ref,
                target=ref.target,
                reason="sensitive -> pinned local",
                fell_back=False,
            )

        # 3. Reason: prefer cloud.
        if kind == Kind.REASON:
            ref = self.registry.get(self.reason_name)
            return Decision(
                model=ref,
                target=ref.target,
                reason="reason -> cloud",
                fell_back=False,
            )

        # 4. Bulk / Auto: prefer local, fall back to cloud if local is down.
        if local_up:
            ref = self.registry.get(self.local_name)
            return Decision(
                model=ref,
                target=ref.target,
                reason="prefer local",
                fell_back=False,
            )
        ref = self.registry.get(self.fallback_name)
        return Decision(
            model=ref,
            target=ref.target,
            reason="local down -> cloud fallback",
            fell_back=True,
        )
