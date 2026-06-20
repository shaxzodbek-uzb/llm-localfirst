"""Tests for the CLI — offline, with the local probe monkeypatched.

No LLM is called: only ``doctor`` and ``route`` are exercised (both make a routing
decision but no completion). Reachability is forced down so the tests never touch a
real socket and the fail-closed path is observable.
"""

from __future__ import annotations

import pytest

from llm_localfirst.cli import main


@pytest.fixture(autouse=True)
def _local_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_localfirst.reachability.Reachability.check",
        lambda self, base_url, *, timeout=2.0: False,
    )


def test_doctor_runs_clean(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Registry (allowlist):" in out
    assert "local" in out and "haiku" in out
    assert "down" in out  # local is forced down


def test_route_bulk_falls_back_to_cloud(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["route", "summarize this", "--kind", "bulk"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "model     = haiku" in out
    assert "target    = cloud" in out
    assert "fell_back = True" in out


def test_route_sensitive_fails_closed_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["route", "redact this record", "--sensitive"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "error:" in captured.err
    assert "fail" in captured.err.lower() or "unreachable" in captured.err.lower()


def test_no_command_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main([])  # argparse requires a subcommand
