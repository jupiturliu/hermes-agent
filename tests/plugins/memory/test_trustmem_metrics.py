"""Unit tests for TrustMem memory scaling metrics."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from plugins.memory.trustmem.metrics import (
    MetricsCollector,
    SessionMetrics,
    TurnMetrics,
    compute_scaling_trend,
    load_metrics_history,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ── TurnMetrics ──────────────────────────────────────────────────────────────


class TestTurnMetrics:
    def test_defaults(self):
        tm = TurnMetrics()
        assert tm.turn_number == 0
        assert tm.prefetch_hit is False
        assert tm.quality_score == 0.0
        assert tm.quality_persisted is False


# ── SessionMetrics ───────────────────────────────────────────────────────────


class TestSessionMetrics:
    def test_hit_rate_no_data(self):
        sm = SessionMetrics()
        assert sm.prefetch_hit_rate == 0.0

    def test_hit_rate(self):
        sm = SessionMetrics(prefetch_hits=3, prefetch_misses=7)
        assert sm.prefetch_hit_rate == 0.3

    def test_filter_rate(self):
        sm = SessionMetrics(episodes_persisted=8, episodes_filtered=2)
        assert sm.filter_rate == 0.2

    def test_filter_rate_no_data(self):
        sm = SessionMetrics()
        assert sm.filter_rate == 0.0

    def test_to_dict(self):
        sm = SessionMetrics(session_id="test", total_turns=5, prefetch_hits=2, prefetch_misses=3)
        d = sm.to_dict()
        assert d["session_id"] == "test"
        assert d["total_turns"] == 5
        assert d["prefetch_hit_rate"] == 0.4
        assert "filter_rate" in d
        assert "outcome_counts" in d


# ── MetricsCollector ─────────────────────────────────────────────────────────


class TestMetricsCollector:
    def test_empty_summarize(self):
        mc = MetricsCollector(session_id="s1", agent_name="hermes")
        sm = mc.summarize()
        assert sm.session_id == "s1"
        assert sm.total_turns == 0
        assert sm.prefetch_hit_rate == 0.0

    def test_record_prefetch(self):
        mc = MetricsCollector()
        mc.record_prefetch(hit=True, latency_ms=5.0)
        mc.record_prefetch(hit=False, latency_ms=50.0,
                          knowledge_results=3, episode_results=2,
                          semantic_entries=2, episodic_entries=3,
                          avg_confidence=0.7)
        sm = mc.summarize()
        assert sm.total_turns == 2
        assert sm.prefetch_hits == 1
        assert sm.prefetch_misses == 1
        assert sm.prefetch_hit_rate == 0.5
        assert sm.avg_prefetch_latency_ms > 0

    def test_record_quality(self):
        mc = MetricsCollector()
        mc.record_prefetch(hit=False, latency_ms=10)  # creates a turn
        mc.record_quality(score=0.6, outcome="success", persisted=True)
        mc.record_prefetch(hit=False, latency_ms=10)
        mc.record_quality(score=0.1, outcome="trivial", persisted=False)

        sm = mc.summarize()
        assert sm.episodes_persisted == 1
        assert sm.episodes_filtered == 1
        assert sm.filter_rate == 0.5
        assert sm.outcome_counts.get("success") == 1
        assert sm.outcome_counts.get("trivial") == 1

    def test_record_distillation(self):
        mc = MetricsCollector()
        mc.record_distillation(knowledge_created=3, episodes_consolidated=15)
        mc.record_distillation(knowledge_created=2, episodes_consolidated=8)

        sm = mc.summarize()
        assert sm.distillation_runs == 2
        assert sm.knowledge_entries_created == 5
        assert sm.episodes_consolidated == 23

    def test_avg_confidence(self):
        mc = MetricsCollector()
        mc.record_prefetch(hit=False, latency_ms=1, avg_confidence=0.8)
        mc.record_prefetch(hit=False, latency_ms=1, avg_confidence=0.6)
        sm = mc.summarize()
        assert sm.avg_confidence == pytest.approx(0.7, abs=0.01)

    def test_persist_creates_file(self, tmp_dir):
        mc = MetricsCollector(session_id="s1", agent_name="hermes")
        mc.record_prefetch(hit=True, latency_ms=5.0)
        mc.record_quality(score=0.7, outcome="success", persisted=True)

        path = mc.persist(tmp_dir)
        assert path is not None
        assert path.exists()

        content = path.read_text()
        data = json.loads(content.strip())
        assert data["session_id"] == "s1"
        assert data["total_turns"] == 1

    def test_persist_appends(self, tmp_dir):
        for i in range(3):
            mc = MetricsCollector(session_id=f"s{i}")
            mc.record_prefetch(hit=bool(i % 2), latency_ms=float(i))
            mc.persist(tmp_dir)

        log_path = tmp_dir / "memory_metrics.jsonl"
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3


# ── load_metrics_history ─────────────────────────────────────────────────────


class TestLoadHistory:
    def test_empty_dir(self, tmp_dir):
        assert load_metrics_history(tmp_dir) == []

    def test_loads_entries(self, tmp_dir):
        log_path = tmp_dir / "memory_metrics.jsonl"
        entries = [
            {"session_id": f"s{i}", "total_turns": i + 1}
            for i in range(5)
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries))

        result = load_metrics_history(tmp_dir)
        assert len(result) == 5
        assert result[0]["session_id"] == "s0"

    def test_respects_limit(self, tmp_dir):
        log_path = tmp_dir / "memory_metrics.jsonl"
        entries = [{"session_id": f"s{i}"} for i in range(20)]
        log_path.write_text("\n".join(json.dumps(e) for e in entries))

        result = load_metrics_history(tmp_dir, limit=5)
        assert len(result) == 5
        assert result[0]["session_id"] == "s15"  # last 5


# ── compute_scaling_trend ────────────────────────────────────────────────────


class TestScalingTrend:
    def test_empty_history(self):
        trend = compute_scaling_trend([])
        assert trend["sessions"] == 0
        assert trend["trend"] == "insufficient_data"

    def test_improving_trend(self):
        history = [
            {"prefetch_hit_rate": 0.1, "filter_rate": 0.5, "avg_confidence": 0.3,
             "episodes_persisted": 5, "knowledge_entries_created": 0},
            {"prefetch_hit_rate": 0.2, "filter_rate": 0.4, "avg_confidence": 0.4,
             "episodes_persisted": 8, "knowledge_entries_created": 1},
            {"prefetch_hit_rate": 0.5, "filter_rate": 0.3, "avg_confidence": 0.5,
             "episodes_persisted": 10, "knowledge_entries_created": 2},
            {"prefetch_hit_rate": 0.7, "filter_rate": 0.2, "avg_confidence": 0.6,
             "episodes_persisted": 12, "knowledge_entries_created": 3},
        ]
        trend = compute_scaling_trend(history)
        assert trend["sessions"] == 4
        assert trend["trend"] == "improving"
        assert trend["total_episodes_persisted"] == 35
        assert trend["total_knowledge_created"] == 6

    def test_stable_trend(self):
        history = [
            {"prefetch_hit_rate": 0.5, "filter_rate": 0.3, "avg_confidence": 0.5,
             "episodes_persisted": 10, "knowledge_entries_created": 1},
        ] * 4
        trend = compute_scaling_trend(history)
        assert trend["trend"] == "stable"

    def test_declining_trend(self):
        history = [
            {"prefetch_hit_rate": 0.8, "filter_rate": 0.1, "avg_confidence": 0.7,
             "episodes_persisted": 15, "knowledge_entries_created": 3},
            {"prefetch_hit_rate": 0.7, "filter_rate": 0.2, "avg_confidence": 0.6,
             "episodes_persisted": 12, "knowledge_entries_created": 2},
            {"prefetch_hit_rate": 0.3, "filter_rate": 0.4, "avg_confidence": 0.4,
             "episodes_persisted": 8, "knowledge_entries_created": 1},
            {"prefetch_hit_rate": 0.1, "filter_rate": 0.5, "avg_confidence": 0.3,
             "episodes_persisted": 5, "knowledge_entries_created": 0},
        ]
        trend = compute_scaling_trend(history)
        assert trend["trend"] == "declining"

    def test_includes_latest_session(self):
        history = [
            {"prefetch_hit_rate": 0.5, "filter_rate": 0.3, "avg_confidence": 0.5,
             "episodes_persisted": 10, "knowledge_entries_created": 1, "session_id": "latest"},
        ]
        trend = compute_scaling_trend(history)
        assert trend["latest_session"]["session_id"] == "latest"
