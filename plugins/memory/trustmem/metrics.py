"""
TrustMem Memory Scaling Metrics
================================
Lightweight observability for memory system performance, inspired by
Databricks MemAlign's memory scaling curve analysis.

Tracks per-turn and session-level metrics:
  - Prefetch: hit/miss rate, latency, result counts
  - Quality judge: accept/reject/borderline rates
  - Working memory: semantic vs episodic slot fill
  - Distillation: episodes processed, knowledge created
  - Search: query counts, avg scores

Metrics are collected in-memory during a session and persisted to a
JSON log at session end for longitudinal analysis.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TurnMetrics:
    """Metrics for a single turn."""
    turn_number: int = 0
    timestamp: str = ""
    # Prefetch
    prefetch_hit: bool = False       # cache had content
    prefetch_latency_ms: float = 0   # time to produce context
    # Search results
    knowledge_results: int = 0       # results from knowledge_search
    episode_results: int = 0         # results from episode recall
    semantic_entries: int = 0        # entries classified as semantic
    episodic_entries: int = 0        # entries classified as episodic
    avg_confidence: float = 0.0      # mean confidence of returned results
    # Quality judge
    quality_score: float = 0.0       # heuristic score for this turn
    quality_outcome: str = ""        # success/partial/failure/trivial
    quality_persisted: bool = False  # whether episode was persisted


@dataclass
class SessionMetrics:
    """Aggregated metrics for a full session."""
    session_id: str = ""
    agent_name: str = ""
    started_at: str = ""
    ended_at: str = ""
    total_turns: int = 0
    # Prefetch
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    avg_prefetch_latency_ms: float = 0.0
    # Search
    total_searches: int = 0
    avg_knowledge_results: float = 0.0
    avg_episode_results: float = 0.0
    avg_semantic_entries: float = 0.0
    avg_episodic_entries: float = 0.0
    avg_confidence: float = 0.0
    # Quality
    episodes_persisted: int = 0
    episodes_filtered: int = 0
    avg_quality_score: float = 0.0
    outcome_counts: dict[str, int] = field(default_factory=dict)
    # Distillation
    distillation_runs: int = 0
    knowledge_entries_created: int = 0
    episodes_consolidated: int = 0

    @property
    def prefetch_hit_rate(self) -> float:
        total = self.prefetch_hits + self.prefetch_misses
        return self.prefetch_hits / total if total > 0 else 0.0

    @property
    def filter_rate(self) -> float:
        total = self.episodes_persisted + self.episodes_filtered
        return self.episodes_filtered / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_turns": self.total_turns,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "prefetch_hit_rate": round(self.prefetch_hit_rate, 3),
            "avg_prefetch_latency_ms": round(self.avg_prefetch_latency_ms, 1),
            "total_searches": self.total_searches,
            "avg_knowledge_results": round(self.avg_knowledge_results, 1),
            "avg_episode_results": round(self.avg_episode_results, 1),
            "avg_semantic_entries": round(self.avg_semantic_entries, 1),
            "avg_episodic_entries": round(self.avg_episodic_entries, 1),
            "avg_confidence": round(self.avg_confidence, 3),
            "episodes_persisted": self.episodes_persisted,
            "episodes_filtered": self.episodes_filtered,
            "filter_rate": round(self.filter_rate, 3),
            "avg_quality_score": round(self.avg_quality_score, 3),
            "outcome_counts": self.outcome_counts,
            "distillation_runs": self.distillation_runs,
            "knowledge_entries_created": self.knowledge_entries_created,
            "episodes_consolidated": self.episodes_consolidated,
        }
        return d


class MetricsCollector:
    """In-memory metrics collector for one session."""

    def __init__(self, session_id: str = "", agent_name: str = ""):
        self._session_id = session_id
        self._agent_name = agent_name
        self._started_at = datetime.now().isoformat()
        self._turns: list[TurnMetrics] = []
        self._current_turn = 0
        # Distillation counters
        self._distillation_runs = 0
        self._knowledge_created = 0
        self._episodes_consolidated = 0
        # Search counters
        self._total_searches = 0
        self._search_knowledge_counts: list[int] = []
        self._search_episode_counts: list[int] = []
        self._search_confidences: list[float] = []

    # ── Per-turn recording ───────────────────────────────────────────────

    def record_prefetch(
        self,
        hit: bool,
        latency_ms: float,
        knowledge_results: int = 0,
        episode_results: int = 0,
        semantic_entries: int = 0,
        episodic_entries: int = 0,
        avg_confidence: float = 0.0,
    ) -> None:
        """Record prefetch metrics for the current turn."""
        self._current_turn += 1
        self._total_searches += 1
        self._search_knowledge_counts.append(knowledge_results)
        self._search_episode_counts.append(episode_results)
        if avg_confidence > 0:
            self._search_confidences.append(avg_confidence)

        tm = TurnMetrics(
            turn_number=self._current_turn,
            timestamp=datetime.now().isoformat(),
            prefetch_hit=hit,
            prefetch_latency_ms=latency_ms,
            knowledge_results=knowledge_results,
            episode_results=episode_results,
            semantic_entries=semantic_entries,
            episodic_entries=episodic_entries,
            avg_confidence=avg_confidence,
        )
        self._turns.append(tm)

    def record_quality(
        self,
        score: float,
        outcome: str,
        persisted: bool,
    ) -> None:
        """Record quality judge verdict for the current turn."""
        if self._turns:
            tm = self._turns[-1]
            tm.quality_score = score
            tm.quality_outcome = outcome
            tm.quality_persisted = persisted

    def record_distillation(
        self,
        knowledge_created: int,
        episodes_consolidated: int,
    ) -> None:
        """Record a distillation run."""
        self._distillation_runs += 1
        self._knowledge_created += knowledge_created
        self._episodes_consolidated += episodes_consolidated

    # ── Session summary ──────────────────────────────────────────────────

    def summarize(self) -> SessionMetrics:
        """Aggregate turn-level metrics into session summary."""
        sm = SessionMetrics(
            session_id=self._session_id,
            agent_name=self._agent_name,
            started_at=self._started_at,
            ended_at=datetime.now().isoformat(),
            total_turns=self._current_turn,
        )

        # Distillation (independent of turns)
        sm.distillation_runs = self._distillation_runs
        sm.knowledge_entries_created = self._knowledge_created
        sm.episodes_consolidated = self._episodes_consolidated

        if not self._turns:
            return sm

        # Prefetch
        hits = [t for t in self._turns if t.prefetch_hit]
        sm.prefetch_hits = len(hits)
        sm.prefetch_misses = len(self._turns) - len(hits)
        latencies = [t.prefetch_latency_ms for t in self._turns if t.prefetch_latency_ms > 0]
        sm.avg_prefetch_latency_ms = sum(latencies) / len(latencies) if latencies else 0

        # Search
        sm.total_searches = self._total_searches
        if self._search_knowledge_counts:
            sm.avg_knowledge_results = sum(self._search_knowledge_counts) / len(self._search_knowledge_counts)
        if self._search_episode_counts:
            sm.avg_episode_results = sum(self._search_episode_counts) / len(self._search_episode_counts)
        if self._search_confidences:
            sm.avg_confidence = sum(self._search_confidences) / len(self._search_confidences)

        # Working memory composition
        semantic_fills = [t.semantic_entries for t in self._turns]
        episodic_fills = [t.episodic_entries for t in self._turns]
        sm.avg_semantic_entries = sum(semantic_fills) / len(semantic_fills) if semantic_fills else 0
        sm.avg_episodic_entries = sum(episodic_fills) / len(episodic_fills) if episodic_fills else 0

        # Quality
        quality_scores = [t.quality_score for t in self._turns if t.quality_score > 0]
        sm.avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        sm.episodes_persisted = sum(1 for t in self._turns if t.quality_persisted)
        sm.episodes_filtered = sum(1 for t in self._turns if t.quality_score > 0 and not t.quality_persisted)
        for t in self._turns:
            if t.quality_outcome:
                sm.outcome_counts[t.quality_outcome] = sm.outcome_counts.get(t.quality_outcome, 0) + 1

        return sm

    # ── Persistence ──────────────────────────────────────────────────────

    def persist(self, metrics_dir: Path) -> Path | None:
        """Append session summary to metrics log file."""
        try:
            metrics_dir.mkdir(parents=True, exist_ok=True)
            log_path = metrics_dir / "memory_metrics.jsonl"
            summary = self.summarize()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary.to_dict(), ensure_ascii=False) + "\n")
            return log_path
        except Exception as exc:
            logger.debug("metrics persist error: %s", exc)
            return None


def load_metrics_history(metrics_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    """Load recent session metrics from the JSONL log."""
    log_path = metrics_dir / "memory_metrics.jsonl"
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        entries = [json.loads(line) for line in lines if line.strip()]
        return entries[-limit:]
    except Exception as exc:
        logger.debug("metrics load error: %s", exc)
        return []


def compute_scaling_trend(history: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyze memory scaling trend from session history.
    Shows how memory effectiveness changes over time.
    """
    if not history:
        return {"sessions": 0, "trend": "insufficient_data"}

    n = len(history)
    hit_rates = [h.get("prefetch_hit_rate", 0) for h in history]
    filter_rates = [h.get("filter_rate", 0) for h in history]
    avg_confidences = [h.get("avg_confidence", 0) for h in history]
    total_persisted = sum(h.get("episodes_persisted", 0) for h in history)
    total_knowledge = sum(h.get("knowledge_entries_created", 0) for h in history)

    # Trend: compare first half vs second half
    mid = max(1, n // 2)
    first_half_hits = hit_rates[:mid]
    second_half_hits = hit_rates[mid:]

    avg_first = sum(first_half_hits) / len(first_half_hits) if first_half_hits else 0
    avg_second = sum(second_half_hits) / len(second_half_hits) if second_half_hits else 0

    if avg_second > avg_first + 0.05:
        trend = "improving"
    elif avg_second < avg_first - 0.05:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "sessions": n,
        "trend": trend,
        "total_episodes_persisted": total_persisted,
        "total_knowledge_created": total_knowledge,
        "avg_prefetch_hit_rate": round(sum(hit_rates) / n, 3) if n else 0,
        "avg_filter_rate": round(sum(filter_rates) / n, 3) if n else 0,
        "avg_confidence": round(sum(avg_confidences) / n, 3) if n else 0,
        "latest_session": history[-1] if history else None,
    }
