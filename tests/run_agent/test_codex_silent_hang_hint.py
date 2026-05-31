"""Tests for the ``_codex_silent_hang_hint`` heuristic.

The helper substitutes an actionable hint into the stale-call timeout
warning when the request matches a known Codex silent-reject pattern
(gpt-5.5 family on the ChatGPT Codex backend).  See issue #21444 for
symptom history.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _make_agent(tmp_path: Path, **overrides):
    from run_agent import AIAgent
    kwargs = dict(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)


@pytest.fixture(autouse=True)
def _isolate_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")


# ── positive cases: hint fires ─────────────────────────────────────────────


def test_hint_fires_for_bare_gpt_5_5_on_codex(tmp_path):
    agent = _make_agent(tmp_path)
    agent.api_mode = "codex_responses"
    hint = agent._codex_silent_hang_hint(model="gpt-5.5")
    assert hint is not None
    assert "gpt-5.4-codex" in hint
    assert "fallback chain" in hint


def test_hint_fires_for_vendor_prefixed_gpt_5_5(tmp_path):
    agent = _make_agent(tmp_path, model="openai/gpt-5.5")
    agent.api_mode = "codex_responses"
    hint = agent._codex_silent_hang_hint(model="openai/gpt-5.5")
    assert hint is not None


def test_hint_fires_for_gpt_5_5_codex_suffix(tmp_path):
    agent = _make_agent(tmp_path, model="gpt-5.5-codex")
    agent.api_mode = "codex_responses"
    hint = agent._codex_silent_hang_hint(model="gpt-5.5-codex")
    assert hint is not None


def test_gpt_5_5_on_codex_is_redirected_so_self_model_hint_is_moot(tmp_path):
    """gpt-5.5 on the Codex backend is now redirected to gpt-5.4-codex at init
    (``_avoid_silently_rejected_codex_model``), a stronger fix than the advisory
    hint. So an agent never *persists* gpt-5.5 as ``self.model`` on codex: the
    no-arg hint (which falls back to ``self.model``) correctly returns None
    because the redirect already removed the silent-reject risk. The hint still
    fires for an explicit ``model="gpt-5.5"`` arg (covered above)."""
    agent = _make_agent(tmp_path)
    agent.api_mode = "codex_responses"
    assert agent.model == "gpt-5.4-codex"  # redirected at init
    assert agent._codex_silent_hang_hint() is None  # nothing left to warn about


# ── negative cases: hint stays None ────────────────────────────────────────


def test_hint_skipped_for_gpt_5_4_codex(tmp_path):
    """gpt-5.4-codex is the recommended workaround — must not trigger."""
    agent = _make_agent(tmp_path, model="gpt-5.4-codex")
    agent.api_mode = "codex_responses"
    assert agent._codex_silent_hang_hint(model="gpt-5.4-codex") is None


def test_hint_skipped_for_gpt_5_50_false_positive(tmp_path):
    """``gpt-5.50`` (hypothetical future SKU) must not regex-match gpt-5.5."""
    agent = _make_agent(tmp_path, model="gpt-5.50")
    agent.api_mode = "codex_responses"
    assert agent._codex_silent_hang_hint(model="gpt-5.50") is None


def test_hint_skipped_for_non_codex_api_mode(tmp_path):
    """Hint only fires on the Codex Responses path."""
    agent = _make_agent(tmp_path)
    agent.api_mode = "chat_completions"
    assert agent._codex_silent_hang_hint(model="gpt-5.5") is None


def test_hint_skipped_for_non_codex_provider(tmp_path):
    """Same model on a non-Codex provider does not trigger."""
    agent = _make_agent(
        tmp_path,
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-5.5",
    )
    agent.api_mode = "codex_responses"
    assert agent._codex_silent_hang_hint(model="openai/gpt-5.5") is None


def test_hint_skipped_for_empty_model(tmp_path):
    """Explicit empty string ``model`` short-circuits the regex."""
    agent = _make_agent(tmp_path, model="gpt-5.4-codex")  # self.model non-matching
    agent.api_mode = "codex_responses"
    # Explicit empty string: regex won't match
    assert agent._codex_silent_hang_hint(model="") is None
    # model=None falls back to self.model which is gpt-5.4-codex, also no match
    assert agent._codex_silent_hang_hint(model=None) is None


def test_hint_skipped_for_unrelated_model_on_codex(tmp_path):
    agent = _make_agent(tmp_path, model="gpt-4-turbo")
    agent.api_mode = "codex_responses"
    assert agent._codex_silent_hang_hint(model="gpt-4-turbo") is None
