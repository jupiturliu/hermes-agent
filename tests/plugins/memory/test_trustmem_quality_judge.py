"""Unit tests for TrustMem episode quality judge."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from plugins.memory.trustmem.quality_judge import (
    QualityVerdict,
    judge_episode,
    score_heuristic,
    score_llm,
    _parse_llm_verdict,
    _signal_content_length,
    _signal_specificity,
    _signal_error_presence,
    _signal_tool_usage,
    _signal_trivial,
    _signal_insight,
    _signal_cjk_content,
)


# ── Individual signal tests ──────────────────────────────────────────────────


class TestContentLength:
    def test_empty(self):
        assert _signal_content_length("", "") == 0.0

    def test_short(self):
        assert _signal_content_length("hi", "hello") == 0.0

    def test_medium(self):
        score = _signal_content_length("x" * 50, "y" * 60)
        assert 0.15 < score < 0.55

    def test_long(self):
        score = _signal_content_length("x" * 500, "y" * 600)
        assert score >= 0.7


class TestSpecificity:
    def test_vague_chat(self):
        score = _signal_specificity("how are you", "I am fine")
        assert score < 0.2

    def test_code_block(self):
        score = _signal_specificity("fix this", "```python\ndef foo(): pass\n```")
        assert score >= 0.3

    def test_file_path(self):
        score = _signal_specificity("edit /src/main.py", "done")
        assert score > 0.0

    def test_technical_terms(self):
        score = _signal_specificity(
            "deploy the api",
            "Updated the docker config for the http json schema deployment"
        )
        assert score >= 0.2


class TestErrorPresence:
    def test_no_errors(self):
        assert _signal_error_presence("All tests passed successfully") == 0.0

    def test_single_error(self):
        score = _signal_error_presence("Got an ImportError when running")
        assert 0.2 < score < 0.5

    def test_many_errors(self):
        text = "Error: failed with ValueError, TypeError, RuntimeError traceback"
        score = _signal_error_presence(text)
        assert score >= 0.5


class TestToolUsage:
    def test_no_tools(self):
        assert _signal_tool_usage("Just chatting") == 0.0

    def test_code_block(self):
        score = _signal_tool_usage("```python\nimport os\n```")
        assert score > 0.0

    def test_file_operations(self):
        score = _signal_tool_usage("Created file src/main.py and modified file README.md")
        assert score >= 0.4


class TestTrivial:
    def test_ok_response(self):
        assert _signal_trivial("ok", "something") >= 0.5

    def test_thanks(self):
        assert _signal_trivial("thanks!", "you're welcome") >= 0.5

    def test_short_both(self):
        assert _signal_trivial("yes", "no") >= 0.5

    def test_substantive(self):
        assert _signal_trivial(
            "How do I configure the database connection?",
            "You need to set the DATABASE_URL environment variable"
        ) == 0.0


class TestInsight:
    def test_no_insight(self):
        assert _signal_insight("Here is the list of files") == 0.0

    def test_reasoning(self):
        score = _signal_insight(
            "The root cause is a race condition. Therefore we should "
            "use a mutex. I recommend adding a lock because the critical "
            "section must be protected."
        )
        assert score >= 0.4

    def test_decision(self):
        score = _signal_insight("The key trade-off here is latency vs throughput")
        assert score > 0.0


class TestCJK:
    def test_no_cjk(self):
        assert _signal_cjk_content("hello", "world") == 0.0

    def test_mixed(self):
        score = _signal_cjk_content("配置数据库", "Set DATABASE_URL")
        assert score > 0.0

    def test_mostly_cjk(self):
        score = _signal_cjk_content("配置数据库连接参数和超时设置", "已完成修改")
        assert score >= 0.1


# ── Heuristic scorer tests ───────────────────────────────────────────────────


class TestScoreHeuristic:
    def test_trivial_exchange(self):
        verdict = score_heuristic("ok", "sure")
        assert verdict.score < 0.2
        assert verdict.outcome == "trivial"

    def test_substantive_code_exchange(self):
        verdict = score_heuristic(
            "Fix the database connection pool in src/db.py",
            "I've modified the config to use connection pooling. "
            "```python\npool = create_pool(max_size=20)\n```\n"
            "The key insight is that we should reuse connections "
            "because creating new ones is expensive."
        )
        assert verdict.score >= 0.3
        assert verdict.outcome == "success"

    def test_error_exchange(self):
        verdict = score_heuristic(
            "Run the tests",
            "Error: ImportError failed with ValueError and TypeError. "
            "Traceback shows RuntimeError in the module. Cannot continue. "
            "Permission denied when accessing the file."
        )
        assert verdict.outcome in ("failure", "partial")

    def test_empty_input(self):
        verdict = score_heuristic("", "")
        assert verdict.score == 0.0

    def test_score_always_in_range(self):
        # Edge cases should never exceed [0, 1]
        for user, asst in [
            ("x" * 10000, "y" * 10000),
            ("", ""),
            ("ok", "ok"),
            ("a", "error " * 100),
        ]:
            v = score_heuristic(user, asst)
            assert 0.0 <= v.score <= 1.0
            assert v.outcome in ("success", "partial", "failure", "trivial")

    def test_signals_dict_populated(self):
        verdict = score_heuristic("test query", "test response with some content")
        assert "content_length" in verdict.signals
        assert "specificity" in verdict.signals
        assert "trivial" in verdict.signals
        assert len(verdict.signals) == 7


# ── LLM judge tests ─────────────────────────────────────────────────────────


class TestParseLLMVerdict:
    def test_valid_json(self):
        response = '{"score": 0.75, "outcome": "success", "reason": "good"}'
        verdict = _parse_llm_verdict(response)
        assert verdict is not None
        assert verdict.score == 0.75
        assert verdict.outcome == "success"

    def test_markdown_fenced(self):
        response = '```json\n{"score": 0.5, "outcome": "partial"}\n```'
        verdict = _parse_llm_verdict(response)
        assert verdict is not None
        assert verdict.score == 0.5

    def test_json_in_text(self):
        response = 'Here is my rating: {"score": 0.8, "outcome": "success"} end.'
        verdict = _parse_llm_verdict(response)
        assert verdict is not None
        assert verdict.score == 0.8

    def test_missing_score(self):
        response = '{"outcome": "success"}'
        assert _parse_llm_verdict(response) is None

    def test_garbage(self):
        assert _parse_llm_verdict("not json") is None

    def test_score_clamped(self):
        response = '{"score": 1.5, "outcome": "success"}'
        verdict = _parse_llm_verdict(response)
        assert verdict is not None
        assert verdict.score == 1.0

    def test_invalid_outcome_defaults(self):
        response = '{"score": 0.5, "outcome": "unknown_value"}'
        verdict = _parse_llm_verdict(response)
        assert verdict is not None
        assert verdict.outcome == "success"


class TestScoreLLM:
    def test_success(self):
        llm_fn = MagicMock(return_value='{"score": 0.8, "outcome": "success"}')
        verdict = score_llm("query", "response", llm_fn)
        assert verdict is not None
        assert verdict.score == 0.8
        llm_fn.assert_called_once()

    def test_llm_failure_returns_none(self):
        llm_fn = MagicMock(side_effect=RuntimeError("API down"))
        verdict = score_llm("query", "response", llm_fn)
        assert verdict is None

    def test_bad_llm_output_returns_none(self):
        llm_fn = MagicMock(return_value="garbage output")
        verdict = score_llm("query", "response", llm_fn)
        assert verdict is None


# ── Combined judge tests ─────────────────────────────────────────────────────


class TestJudgeEpisode:
    def test_high_quality_persists(self):
        verdict = judge_episode(
            "Implement the authentication middleware for the API gateway",
            "I've created the middleware in ```python\n"
            "class AuthMiddleware:\n    def __init__(self): pass\n```\n"
            "The key decision is to validate tokens because "
            "we must prevent unauthorized access. "
            "Created file src/middleware/auth.py",
            threshold=0.3,
        )
        assert verdict.should_persist is True
        assert verdict.score >= 0.3

    def test_trivial_filtered(self):
        verdict = judge_episode("ok", "sure", threshold=0.3)
        assert verdict.should_persist is False
        assert verdict.score < 0.3

    def test_threshold_respected(self):
        # Same content, different thresholds
        v_low = judge_episode("test something", "done testing", threshold=0.01)
        v_high = judge_episode("test something", "done testing", threshold=0.99)
        assert v_low.should_persist is True
        assert v_high.should_persist is False

    def test_llm_called_for_borderline(self):
        """LLM should be called when score is in borderline range."""
        llm_fn = MagicMock(return_value='{"score": 0.7, "outcome": "success"}')

        # Build input that lands in borderline range (0.25–0.45)
        verdict = judge_episode(
            "check the logs",
            "I looked at the log file and found some entries",
            threshold=0.3,
            llm_fn=llm_fn,
        )

        # Whether LLM was called depends on heuristic score
        heuristic = score_heuristic(
            "check the logs",
            "I looked at the log file and found some entries",
        )
        if 0.25 <= heuristic.score <= 0.45:
            llm_fn.assert_called_once()
        # Either way, verdict should be valid
        assert 0.0 <= verdict.score <= 1.0

    def test_llm_not_called_for_clear_reject(self):
        """LLM should NOT be called when heuristic clearly rejects."""
        llm_fn = MagicMock(return_value='{"score": 0.9}')
        judge_episode("ok", "sure", threshold=0.3, llm_fn=llm_fn)
        llm_fn.assert_not_called()

    def test_llm_not_called_for_clear_accept(self):
        """LLM should NOT be called when heuristic clearly accepts."""
        llm_fn = MagicMock(return_value='{"score": 0.9}')
        judge_episode(
            "Implement the full database migration with schema changes",
            "```python\n" + "x = 1\n" * 50 + "```\n"
            "The critical decision here is to use transactions because "
            "we must maintain data integrity. Created file migrations/001.py "
            "and modified file config/database.yaml",
            threshold=0.3,
            llm_fn=llm_fn,
        )
        llm_fn.assert_not_called()

    def test_no_llm_fn_is_fine(self):
        """Should work without LLM function."""
        verdict = judge_episode(
            "How does the cache invalidation work?",
            "The cache uses TTL-based expiration with a 5 minute default",
            threshold=0.3,
            llm_fn=None,
        )
        assert isinstance(verdict, QualityVerdict)
        assert 0.0 <= verdict.score <= 1.0
