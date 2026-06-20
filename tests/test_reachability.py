"""Tests for reachability.py — the cached, never-raising local probe (SPEC case 7).

No real network: ``urllib.request.urlopen`` is monkeypatched so the suite stays offline.
"""

from __future__ import annotations

import urllib.error

import pytest

from llm_localfirst import Reachability

BASE = "http://localhost:11434/v1"


class _FakeResp:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_none_base_url_is_down() -> None:
    r = Reachability()
    assert r.check(None) is False
    assert r.check("") is False


def test_probe_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_urlopen(req, timeout=2.0):  # noqa: ANN001 - test shim
        calls["n"] += 1
        return _FakeResp(200)

    monkeypatch.setattr("llm_localfirst.reachability.urllib.request.urlopen", fake_urlopen)
    r = Reachability(ttl=100.0)
    assert r.check(BASE) is True
    assert calls["n"] == 1


def test_result_is_cached_within_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_urlopen(req, timeout=2.0):  # noqa: ANN001 - test shim
        calls["n"] += 1
        return _FakeResp(200)

    monkeypatch.setattr("llm_localfirst.reachability.urllib.request.urlopen", fake_urlopen)
    r = Reachability(ttl=100.0)
    assert r.check(BASE) is True
    assert r.check(BASE) is True
    assert calls["n"] == 1, "second call within TTL must use the cache, not re-probe"


def test_clear_forces_reprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_urlopen(req, timeout=2.0):  # noqa: ANN001 - test shim
        calls["n"] += 1
        return _FakeResp(200)

    monkeypatch.setattr("llm_localfirst.reachability.urllib.request.urlopen", fake_urlopen)
    r = Reachability(ttl=100.0)
    r.check(BASE)
    r.clear()
    r.check(BASE)
    assert calls["n"] == 2


def test_connection_error_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(req, timeout=2.0):  # noqa: ANN001 - test shim
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("llm_localfirst.reachability.urllib.request.urlopen", boom)
    r = Reachability()
    assert r.check(BASE) is False


def test_non_2xx_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req, timeout=2.0):  # noqa: ANN001 - test shim
        return _FakeResp(503)

    monkeypatch.setattr("llm_localfirst.reachability.urllib.request.urlopen", fake_urlopen)
    r = Reachability()
    assert r.check(BASE) is False


def test_distinct_urls_cached_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_urlopen(req, timeout=2.0):  # noqa: ANN001 - test shim
        seen.append(req.full_url)
        return _FakeResp(200)

    monkeypatch.setattr("llm_localfirst.reachability.urllib.request.urlopen", fake_urlopen)
    r = Reachability(ttl=100.0)
    r.check("http://a:1/v1")
    r.check("http://b:2/v1")
    assert len(seen) == 2
