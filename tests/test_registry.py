"""Tests for registry.py — the allowlist (SSRF / cost guard).

SPEC test case 1: allowlist contains/raises; ModelNotAllowed for unknown; by_target.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from llm_localfirst import ModelNotAllowed, ModelRef, Registry, default_registry


def test_contains_known_and_unknown(registry: Registry) -> None:
    assert "local" in registry
    assert "haiku" in registry
    assert "totally-unknown" not in registry


def test_is_allowed(registry: Registry) -> None:
    assert registry.is_allowed("local") is True
    assert registry.is_allowed("gpt") is True
    assert registry.is_allowed("nope") is False
    # None is never allowlisted (it means "no explicit override").
    assert registry.is_allowed(None) is False


def test_get_returns_modelref(registry: Registry) -> None:
    ref = registry.get("local")
    assert isinstance(ref, ModelRef)
    assert ref.name == "local"
    assert ref.target == "local"
    assert ref.provider == "openai_compat"
    assert ref.model_id == "qwen2.5:7b"


def test_get_unknown_raises_model_not_allowed(registry: Registry) -> None:
    with pytest.raises(ModelNotAllowed):
        registry.get("definitely-not-here")


def test_names_lists_all_keys(registry: Registry) -> None:
    names = registry.names()
    assert set(names) == {"local", "haiku", "sonnet", "opus", "gpt"}


def test_by_target_splits_local_and_cloud(registry: Registry) -> None:
    local = registry.by_target("local")
    cloud = registry.by_target("cloud")

    assert [m.name for m in local] == ["local"]
    assert {m.name for m in cloud} == {"haiku", "sonnet", "opus", "gpt"}
    # Every returned ref actually has the requested target.
    assert all(m.target == "local" for m in local)
    assert all(m.target == "cloud" for m in cloud)


def test_modelref_is_frozen() -> None:
    ref = ModelRef(name="x", target="local", provider="openai_compat", model_id="m")
    with pytest.raises(FrozenInstanceError):
        ref.name = "y"  # type: ignore[misc]


def test_modelref_optional_fields_default_none() -> None:
    ref = ModelRef(name="x", target="cloud", provider="anthropic", model_id="claude")
    assert ref.base_url is None
    assert ref.api_key_env is None


# --------------------------------------------------------------------------------------
# default_registry(): built from Settings, must contain the documented baseline keys.
# --------------------------------------------------------------------------------------


def test_default_registry_has_baseline_names() -> None:
    reg = default_registry()
    names = set(reg.names())
    # Always present: one local model + the three Claude models.
    assert {"local", "opus", "sonnet", "haiku"} <= names


def test_default_registry_claude_model_ids() -> None:
    reg = default_registry()
    assert reg.get("opus").model_id == "claude-opus-4-8"
    assert reg.get("sonnet").model_id == "claude-sonnet-4-6"
    assert reg.get("haiku").model_id == "claude-haiku-4-5"
    for key in ("opus", "sonnet", "haiku"):
        assert reg.get(key).provider == "anthropic"
        assert reg.get(key).target == "cloud"


def test_default_registry_local_is_local_target() -> None:
    reg = default_registry()
    local = reg.get("local")
    assert local.target == "local"
    assert local.provider == "openai_compat"
    assert local.base_url is not None
