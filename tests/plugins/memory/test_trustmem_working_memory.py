"""Unit tests for TrustMem working memory assembler."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from plugins.memory.trustmem.working_memory import (
    MemoryEntry,
    WorkingMemory,
    assemble_working_memory,
    classify_episode,
    classify_result,
    format_working_memory,
)


# ── classify_result ──────────────────────────────────────────────────────────


class TestClassifyResult:
    def test_peer_reviewed_high_confidence_is_semantic(self):
        r = {"verification_status": "peer_reviewed", "effective_confidence": 0.8}
        assert classify_result(r) == "semantic"

    def test_auto_distilled_is_semantic(self):
        r = {"verification_status": "auto_distilled", "effective_confidence": 0.5}
        assert classify_result(r) == "semantic"

    def test_unverified_is_episodic(self):
        r = {"verification_status": "unverified", "effective_confidence": 0.9}
        assert classify_result(r) == "episodic"

    def test_low_confidence_peer_reviewed_is_episodic(self):
        r = {"verification_status": "peer_reviewed", "effective_confidence": 0.2}
        assert classify_result(r) == "episodic"

    def test_explicit_memory_type_semantic(self):
        r = {"memory_type": "semantic", "verification_status": "unverified", "effective_confidence": 0.1}
        assert classify_result(r) == "semantic"

    def test_explicit_memory_type_episodic(self):
        r = {"memory_type": "episodic", "verification_status": "peer_reviewed", "effective_confidence": 0.9}
        assert classify_result(r) == "episodic"

    def test_empty_result(self):
        assert classify_result({}) == "episodic"

    def test_verified_status(self):
        r = {"verification_status": "verified", "effective_confidence": 0.6}
        assert classify_result(r) == "semantic"


# ── classify_episode ─────────────────────────────────────────────────────────


class TestClassifyEpisode:
    def test_basic_episode(self):
        ep = {"task": "Analyzed GPU performance", "approach": "profiling", "notes": "found bottleneck", "quality_score": 0.8}
        entry = classify_episode(ep)
        assert entry.layer == "episodic"
        assert "GPU performance" in entry.title
        assert entry.confidence == 0.8

    def test_empty_episode(self):
        entry = classify_episode({})
        assert entry.layer == "episodic"
        assert entry.score == 0.5  # default quality


# ── assemble_working_memory ──────────────────────────────────────────────────


class TestAssembleWorkingMemory:
    def _make_knowledge_results(self):
        return [
            {
                "title": "GPU Memory Best Practices",
                "body": "Always pre-allocate CUDA memory pools",
                "score": 10.5,
                "effective_confidence": 0.85,
                "verification_status": "peer_reviewed",
                "domain": "ai-infra",
                "file": "ai-infra/gpu-best-practices.md",
                "age_days": 5,
            },
            {
                "title": "NVLink Configuration Guide",
                "body": "NVLink requires specific topology setup",
                "score": 8.2,
                "effective_confidence": 0.7,
                "verification_status": "auto_distilled",
                "domain": "ai-infra",
                "file": "distilled/nvlink-guide.md",
                "age_days": 2,
            },
            {
                "title": "Yesterday chat about GPUs",
                "body": "We discussed GPU allocation",
                "score": 5.1,
                "effective_confidence": 0.3,
                "verification_status": "unverified",
                "domain": "ai-infra",
                "file": "notes/gpu-chat.md",
                "age_days": 1,
            },
        ]

    def _make_episode_results(self):
        return [
            ({"task": "Profiled CUDA memory usage", "approach": "nvprof", "notes": "Found 2GB leak", "quality_score": 0.9, "id": "ep1"}, 0.85),
            ({"task": "Deployed model to cluster", "approach": "k8s", "notes": "Successful rollout", "quality_score": 0.7, "id": "ep2"}, 0.6),
        ]

    def test_separates_semantic_and_episodic(self):
        wm = assemble_working_memory(
            query="GPU memory",
            knowledge_results=self._make_knowledge_results(),
        )
        # peer_reviewed + auto_distilled -> semantic; unverified -> episodic
        assert len(wm.semantic) == 2
        assert len(wm.episodic) == 1
        assert wm.semantic[0].title == "GPU Memory Best Practices"

    def test_includes_episodes(self):
        wm = assemble_working_memory(
            query="GPU profiling",
            episode_results=self._make_episode_results(),
        )
        assert len(wm.episodic) == 2
        assert len(wm.semantic) == 0

    def test_combined_knowledge_and_episodes(self):
        wm = assemble_working_memory(
            query="GPU memory optimization",
            knowledge_results=self._make_knowledge_results(),
            episode_results=self._make_episode_results(),
        )
        assert len(wm.semantic) >= 1
        assert len(wm.episodic) >= 2  # 1 from knowledge + 2 from episodes
        assert wm.total_entries >= 3

    def test_respects_slot_limits(self):
        wm = assemble_working_memory(
            query="test",
            knowledge_results=self._make_knowledge_results(),
            episode_results=self._make_episode_results(),
            semantic_slots=1,
            episodic_slots=1,
        )
        assert len(wm.semantic) <= 1
        assert len(wm.episodic) <= 1

    def test_deduplicates_by_title(self):
        duped = [
            {"title": "Same Title", "body": "A", "score": 10, "effective_confidence": 0.8,
             "verification_status": "peer_reviewed"},
            {"title": "Same Title", "body": "B", "score": 8, "effective_confidence": 0.7,
             "verification_status": "peer_reviewed"},
        ]
        wm = assemble_working_memory(query="test", knowledge_results=duped)
        assert wm.total_entries == 1

    def test_empty_inputs(self):
        wm = assemble_working_memory(query="test")
        assert wm.is_empty
        assert wm.total_entries == 0

    def test_none_inputs(self):
        wm = assemble_working_memory(
            query="test",
            knowledge_results=None,
            episode_results=None,
        )
        assert wm.is_empty

    def test_sorted_by_score(self):
        results = [
            {"title": "Low", "body": "x", "score": 1, "effective_confidence": 0.5,
             "verification_status": "peer_reviewed"},
            {"title": "High", "body": "y", "score": 20, "effective_confidence": 0.9,
             "verification_status": "peer_reviewed"},
            {"title": "Mid", "body": "z", "score": 10, "effective_confidence": 0.7,
             "verification_status": "peer_reviewed"},
        ]
        wm = assemble_working_memory(query="test", knowledge_results=results)
        scores = [e.score for e in wm.semantic]
        assert scores == sorted(scores, reverse=True)


# ── format_working_memory ────────────────────────────────────────────────────


class TestFormatWorkingMemory:
    def test_empty_returns_empty_string(self):
        wm = WorkingMemory()
        assert format_working_memory(wm) == ""

    def test_semantic_only(self):
        wm = WorkingMemory(
            semantic=[
                MemoryEntry(
                    title="Always use connection pools",
                    snippet="Reuse database connections to avoid overhead",
                    score=10.0,
                    confidence=0.9,
                    layer="semantic",
                    verification="peer_reviewed",
                ),
            ],
        )
        output = format_working_memory(wm)
        assert "## Recalled Context" in output
        assert "### Guidelines" in output
        assert "Always use connection pools" in output
        assert "[peer_reviewed]" in output
        assert "### Relevant History" not in output

    def test_episodic_only(self):
        wm = WorkingMemory(
            episodic=[
                MemoryEntry(
                    title="Debugged memory leak",
                    snippet="Found leak in CUDA allocator",
                    score=0.8,
                    confidence=0.7,
                    layer="episodic",
                ),
            ],
        )
        output = format_working_memory(wm)
        assert "### Relevant History" in output
        assert "Debugged memory leak" in output
        assert "### Guidelines" not in output

    def test_both_layers(self):
        wm = WorkingMemory(
            semantic=[
                MemoryEntry(title="Rule A", snippet="Do X", score=10, confidence=0.9,
                           layer="semantic", verification="auto_distilled"),
            ],
            episodic=[
                MemoryEntry(title="Task B", snippet="Did Y", score=5, confidence=0.6,
                           layer="episodic"),
            ],
        )
        output = format_working_memory(wm)
        assert "### Guidelines" in output
        assert "### Relevant History" in output
        # Guidelines should come before History
        guideline_pos = output.index("### Guidelines")
        history_pos = output.index("### Relevant History")
        assert guideline_pos < history_pos

    def test_includes_confidence(self):
        wm = WorkingMemory(
            semantic=[
                MemoryEntry(title="Test", snippet="Content", score=5, confidence=0.75,
                           layer="semantic"),
            ],
        )
        output = format_working_memory(wm)
        assert "0.75" in output

    def test_instruction_text_present(self):
        wm = WorkingMemory(
            semantic=[MemoryEntry(title="R", snippet="S", score=1, confidence=0.5, layer="semantic")],
            episodic=[MemoryEntry(title="E", snippet="P", score=1, confidence=0.5, layer="episodic")],
        )
        output = format_working_memory(wm)
        assert "follow these" in output.lower()
        assert "reference" in output.lower()
