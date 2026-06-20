"""Tests for policy.py — the pure routing brain and the headline privacy guarantee.

Covers SPEC test cases 2-6:
  2. sensitive fail-closed
  3. sensitive never resolves to cloud (the headline invariant)
  4. bulk fallback
  5. reason -> cloud
  6. explicit override + the privacy guard
Plus the structural guard that a mis-wired local_name cannot leak a sensitive call.
"""

from __future__ import annotations

import pytest

from llm_localfirst import (
    Kind,
    LocalUnavailable,
    ModelNotAllowed,
    Policy,
    PrivacyViolation,
    Registry,
)

# --- Case 2: sensitive fail-closed -----------------------------------------------------


def test_sensitive_local_up_pins_local(policy: Policy) -> None:
    d = policy.decide(sensitive=True, kind=Kind.AUTO, model=None, local_up=True)
    assert d.target == "local"
    assert d.model.name == "local"
    assert d.fell_back is False


def test_sensitive_local_down_fails_closed(policy: Policy) -> None:
    with pytest.raises(LocalUnavailable):
        policy.decide(sensitive=True, kind=Kind.AUTO, model=None, local_up=False)


# --- Case 3: sensitive NEVER resolves to cloud (the invariant) --------------------------


def test_sensitive_never_yields_cloud(policy: Policy) -> None:
    """For sensitive=True there is NO input (any kind, local up or down) that returns
    a cloud target. It either pins local or raises — it never leaks to the cloud."""
    for kind in Kind:
        for local_up in (True, False):
            try:
                d = policy.decide(
                    sensitive=True, kind=kind, model=None, local_up=local_up
                )
            except LocalUnavailable:
                continue  # failing closed is the correct non-leaking outcome
            assert d.target == "local", f"leaked to cloud for kind={kind} up={local_up}"


def test_sensitive_never_cloud_even_when_fail_open(registry: Registry) -> None:
    """With sensitive_fail_closed=False the call no longer raises when local is down,
    but it STILL pins local — it must never silently divert sensitive data to cloud."""
    p = Policy(registry, sensitive_fail_closed=False)
    for kind in Kind:
        for local_up in (True, False):
            d = p.decide(sensitive=True, kind=kind, model=None, local_up=local_up)
            assert d.target == "local"


# --- Case 4: bulk fallback -------------------------------------------------------------


def test_bulk_local_up_uses_local(policy: Policy) -> None:
    d = policy.decide(sensitive=False, kind=Kind.BULK, model=None, local_up=True)
    assert d.target == "local"
    assert d.fell_back is False


def test_bulk_local_down_falls_back_to_cloud(policy: Policy) -> None:
    d = policy.decide(sensitive=False, kind=Kind.BULK, model=None, local_up=False)
    assert d.target == "cloud"
    assert d.model.name == "haiku"  # the configured fallback
    assert d.fell_back is True


def test_auto_behaves_like_bulk(policy: Policy) -> None:
    up = policy.decide(sensitive=False, kind=Kind.AUTO, model=None, local_up=True)
    down = policy.decide(sensitive=False, kind=Kind.AUTO, model=None, local_up=False)
    assert up.target == "local"
    assert down.target == "cloud" and down.fell_back is True


# --- Case 5: reason -> cloud -----------------------------------------------------------


def test_reason_goes_cloud_regardless_of_local(policy: Policy) -> None:
    for local_up in (True, False):
        d = policy.decide(sensitive=False, kind=Kind.REASON, model=None, local_up=local_up)
        assert d.target == "cloud"
        assert d.model.name == "haiku"  # the configured reason model
        assert d.fell_back is False


# --- Case 6: explicit override + privacy guard -----------------------------------------


def test_override_uses_named_model(policy: Policy) -> None:
    d = policy.decide(sensitive=False, kind=Kind.AUTO, model="sonnet", local_up=True)
    assert d.model.name == "sonnet"
    assert d.target == "cloud"
    assert d.fell_back is False


def test_override_local_model_allowed(policy: Policy) -> None:
    d = policy.decide(sensitive=False, kind=Kind.REASON, model="local", local_up=False)
    assert d.model.name == "local"
    assert d.target == "local"


def test_override_unknown_model_raises(policy: Policy) -> None:
    with pytest.raises(ModelNotAllowed):
        policy.decide(sensitive=False, kind=Kind.AUTO, model="evil-model", local_up=True)


def test_override_arbitrary_url_string_blocked(policy: Policy) -> None:
    """The allowlist is the SSRF guard: a raw endpoint string is not a model name."""
    with pytest.raises(ModelNotAllowed):
        policy.decide(
            sensitive=False, kind=Kind.AUTO, model="http://evil/v1", local_up=True
        )


def test_sensitive_override_to_cloud_raises_privacy(policy: Policy) -> None:
    for cloud_model in ("haiku", "sonnet", "opus", "gpt"):
        with pytest.raises(PrivacyViolation):
            policy.decide(
                sensitive=True, kind=Kind.AUTO, model=cloud_model, local_up=True
            )


def test_sensitive_override_to_local_allowed(policy: Policy) -> None:
    d = policy.decide(sensitive=True, kind=Kind.AUTO, model="local", local_up=True)
    assert d.target == "local"


# --- Structural guard: a mis-wired "local" key cannot leak a sensitive call -------------


def test_sensitive_misconfigured_local_key_raises_privacy(registry: Registry) -> None:
    """If local_name points at a cloud model (misconfig), a sensitive call must refuse
    (PrivacyViolation), not return a cloud target."""
    p = Policy(registry, local_name="haiku")  # 'haiku' is a cloud model
    with pytest.raises(PrivacyViolation):
        p.decide(sensitive=True, kind=Kind.AUTO, model=None, local_up=True)
