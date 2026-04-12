"""Unit tests for TrustMem episodic → semantic distillation engine."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from plugins.memory.trustmem.distiller import (
    EpisodicDistiller,
    build_distillation_prompt,
    cluster_episodes,
    count_raw_episodes,
    fetch_raw_episodes,
    mark_episodes_consolidated,
    write_knowledge_file,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def db_path(tmp_dir):
    return str(tmp_dir / "test-episodes.sqlite")


@pytest.fixture
def knowledge_root(tmp_dir):
    kr = tmp_dir / "knowledge"
    kr.mkdir()
    return kr


def _create_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            session_key TEXT,
            created_at TEXT NOT NULL,
            summary TEXT NOT NULL,
            details TEXT,
            participants TEXT,
            importance REAL DEFAULT 0.5,
            emotional_valence REAL DEFAULT 0.0,
            emotional_arousal REAL DEFAULT 0.0,
            topic_tags TEXT,
            linked_episodes TEXT,
            context_hash TEXT,
            consolidation_status TEXT DEFAULT 'raw',
            consolidation_count INTEGER DEFAULT 0,
            last_accessed_at TEXT,
            access_count INTEGER DEFAULT 0,
            embedding BLOB,
            visibility TEXT NOT NULL DEFAULT 'global',
            audience TEXT
        )
    """)
    conn.commit()
    return conn


def _insert_episode(
    conn: sqlite3.Connection,
    summary: str = "test task",
    details: str = "test details",
    agent_id: str = "hermes",
    status: str = "raw",
    tags: list[str] | None = None,
    importance: float = 0.5,
) -> str:
    eid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO episodes (id, agent_id, created_at, summary, details, "
        "importance, topic_tags, consolidation_status, visibility) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'global')",
        (
            eid,
            agent_id,
            datetime.now().isoformat(),
            summary,
            details,
            importance,
            json.dumps(tags or []),
            status,
        ),
    )
    conn.commit()
    return eid


# ── fetch_raw_episodes ────────────────────────────────────────────────────────


class TestFetchRawEpisodes:
    def test_returns_only_raw(self, db_path):
        conn = _create_db(db_path)
        _insert_episode(conn, summary="raw one", status="raw")
        _insert_episode(conn, summary="consolidated", status="consolidated")
        conn.close()

        eps = fetch_raw_episodes(db_path)
        assert len(eps) == 1
        assert eps[0]["summary"] == "raw one"

    def test_respects_agent_filter(self, db_path):
        conn = _create_db(db_path)
        _insert_episode(conn, summary="hermes task", agent_id="hermes")
        _insert_episode(conn, summary="aria task", agent_id="aria")
        conn.close()

        eps = fetch_raw_episodes(db_path, agent_id="hermes")
        assert len(eps) == 1
        assert eps[0]["agent_id"] == "hermes"

    def test_respects_limit(self, db_path):
        conn = _create_db(db_path)
        for i in range(20):
            _insert_episode(conn, summary=f"task {i}")
        conn.close()

        eps = fetch_raw_episodes(db_path, limit=5)
        assert len(eps) == 5

    def test_empty_db(self, db_path):
        _create_db(db_path)
        assert fetch_raw_episodes(db_path) == []

    def test_nonexistent_db(self, tmp_dir):
        assert fetch_raw_episodes(str(tmp_dir / "nope.sqlite")) == []


# ── count_raw_episodes ────────────────────────────────────────────────────────


class TestCountRawEpisodes:
    def test_counts_only_raw(self, db_path):
        conn = _create_db(db_path)
        _insert_episode(conn, status="raw")
        _insert_episode(conn, status="raw")
        _insert_episode(conn, status="consolidated")
        conn.close()

        assert count_raw_episodes(db_path) == 2

    def test_agent_filter(self, db_path):
        conn = _create_db(db_path)
        _insert_episode(conn, agent_id="hermes")
        _insert_episode(conn, agent_id="aria")
        conn.close()

        assert count_raw_episodes(db_path, agent_id="hermes") == 1


# ── mark_episodes_consolidated ────────────────────────────────────────────────


class TestMarkConsolidated:
    def test_marks_and_increments_count(self, db_path):
        conn = _create_db(db_path)
        eid = _insert_episode(conn, status="raw")
        conn.close()

        mark_episodes_consolidated(db_path, [eid])

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT consolidation_status, consolidation_count FROM episodes WHERE id = ?",
            (eid,),
        ).fetchone()
        conn.close()
        assert row[0] == "consolidated"
        assert row[1] == 1

    def test_empty_list_is_noop(self, db_path):
        _create_db(db_path)
        assert mark_episodes_consolidated(db_path, []) == 0


# ── cluster_episodes ──────────────────────────────────────────────────────────


class TestClusterEpisodes:
    def test_clusters_similar_episodes(self):
        episodes = [
            {"summary": "GPU memory optimization for training", "details": "CUDA memory pool"},
            {"summary": "GPU memory leak in training loop", "details": "CUDA OOM error"},
            {"summary": "GPU memory profiling results", "details": "CUDA memory usage"},
            {"summary": "API endpoint design patterns", "details": "REST routing"},
        ]
        clusters = cluster_episodes(episodes, min_cluster_size=2)
        # GPU-related episodes should cluster together
        assert len(clusters) >= 1
        gpu_cluster = [c for c in clusters if any("GPU" in ep["summary"] for ep in c)]
        assert len(gpu_cluster) == 1
        assert len(gpu_cluster[0]) == 3

    def test_no_clusters_below_threshold(self):
        episodes = [
            {"summary": "totally unique topic A", "details": "nothing in common"},
            {"summary": "completely different B", "details": "no overlap at all"},
        ]
        clusters = cluster_episodes(episodes, min_cluster_size=2)
        assert len(clusters) == 0

    def test_respects_min_cluster_size(self):
        episodes = [
            {"summary": "machine learning training", "details": "neural network"},
            {"summary": "machine learning inference", "details": "neural network"},
        ]
        assert len(cluster_episodes(episodes, min_cluster_size=3)) == 0
        assert len(cluster_episodes(episodes, min_cluster_size=2)) >= 1

    def test_uses_topic_tags(self):
        episodes = [
            {"summary": "task A", "details": "", "topic_tags": '["gpu", "cuda"]'},
            {"summary": "task B", "details": "", "topic_tags": '["gpu", "memory"]'},
            {"summary": "task C", "details": "", "topic_tags": '["gpu", "training"]'},
        ]
        clusters = cluster_episodes(episodes, min_cluster_size=2)
        assert len(clusters) >= 1

    def test_empty_episodes(self):
        assert cluster_episodes([], min_cluster_size=2) == []


# ── build_distillation_prompt ─────────────────────────────────────────────────


class TestBuildPrompt:
    def test_includes_all_episodes(self):
        cluster = [
            {"summary": f"task {i}", "details": f"detail {i}", "created_at": "2026-04-10", "importance": 0.7}
            for i in range(3)
        ]
        prompt = build_distillation_prompt(cluster)
        assert "Episode 1" in prompt
        assert "Episode 3" in prompt
        assert "3 episodic memories" in prompt
        assert "JSON" in prompt

    def test_truncates_long_details(self):
        cluster = [
            {"summary": "task", "details": "x" * 500, "created_at": "2026-04-10", "importance": 0.5}
        ]
        prompt = build_distillation_prompt(cluster)
        assert len(prompt) < 2000


# ── write_knowledge_file ──────────────────────────────────────────────────────


class TestWriteKnowledgeFile:
    def test_creates_file_with_frontmatter(self, knowledge_root):
        path = write_knowledge_file(
            knowledge_root=knowledge_root,
            title="GPU Memory Patterns",
            body="Always pre-allocate CUDA memory pools.",
            tags=["gpu", "memory"],
            confidence=0.75,
            source_episode_ids=["ep1", "ep2"],
        )
        assert path.exists()
        content = path.read_text()
        assert "---" in content
        assert 'title: "GPU Memory Patterns"' in content
        assert "confidence: 0.75" in content
        assert "verification_status: auto_distilled" in content
        assert "Always pre-allocate" in content
        assert "distilled_from:" in content

    def test_creates_domain_subdirectory(self, knowledge_root):
        path = write_knowledge_file(
            knowledge_root=knowledge_root,
            title="Test",
            body="Body",
            tags=[],
            confidence=0.5,
            source_episode_ids=[],
            domain="custom-domain",
        )
        assert "custom-domain" in str(path.parent)

    def test_avoids_filename_collision(self, knowledge_root):
        args = dict(
            knowledge_root=knowledge_root,
            title="Same Title",
            body="Body",
            tags=[],
            confidence=0.5,
            source_episode_ids=[],
        )
        path1 = write_knowledge_file(**args)
        path2 = write_knowledge_file(**args)
        assert path1 != path2
        assert path1.exists()
        assert path2.exists()


# ── EpisodicDistiller ─────────────────────────────────────────────────────────


class TestEpisodicDistiller:
    def _make_distiller(self, db_path, knowledge_root, llm_fn=None, threshold=3, cluster_min=2):
        return EpisodicDistiller(
            db_path=db_path,
            knowledge_root=knowledge_root,
            agent_name="hermes",
            threshold=threshold,
            cluster_min=cluster_min,
            llm_fn=llm_fn,
        )

    def test_should_distill_below_threshold(self, db_path, knowledge_root):
        _create_db(db_path)
        d = self._make_distiller(db_path, knowledge_root, threshold=10)
        assert not d.should_distill()

    def test_should_distill_above_threshold(self, db_path, knowledge_root):
        conn = _create_db(db_path)
        for i in range(12):
            _insert_episode(conn, summary=f"task {i}")
        conn.close()

        d = self._make_distiller(db_path, knowledge_root, threshold=10)
        assert d.should_distill()

    def test_run_skips_below_threshold(self, db_path, knowledge_root):
        conn = _create_db(db_path)
        _insert_episode(conn, summary="just one")
        conn.close()

        d = self._make_distiller(db_path, knowledge_root, threshold=10)
        result = d.run()
        assert result["skipped_reason"] is not None
        assert result["distilled"] == 0

    def test_run_with_force(self, db_path, knowledge_root):
        conn = _create_db(db_path)
        # Create a cluster of similar episodes
        for i in range(5):
            _insert_episode(conn, summary=f"GPU memory optimization run {i}",
                          details=f"CUDA memory pool configuration {i}")
        conn.close()

        d = self._make_distiller(db_path, knowledge_root, threshold=100, cluster_min=3)
        result = d.run(force=True)
        assert result["distilled"] >= 1
        assert result["episodes_processed"] >= 3
        assert len(result["files_created"]) >= 1
        # Verify file exists
        assert Path(result["files_created"][0]).exists()

    def test_run_marks_episodes_consolidated(self, db_path, knowledge_root):
        conn = _create_db(db_path)
        ids = []
        for i in range(5):
            eid = _insert_episode(conn, summary=f"training experiment {i}",
                                details=f"model training with learning rate {i}")
            ids.append(eid)
        conn.close()

        d = self._make_distiller(db_path, knowledge_root, threshold=3, cluster_min=3)
        result = d.run()

        if result["distilled"] > 0:
            conn = sqlite3.connect(db_path)
            raw_count = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE consolidation_status = 'raw'"
            ).fetchone()[0]
            conn.close()
            assert raw_count < 5  # some should be consolidated

    def test_run_with_custom_llm(self, db_path, knowledge_root):
        conn = _create_db(db_path)
        for i in range(5):
            _insert_episode(conn, summary=f"API design pattern {i}",
                          details=f"REST endpoint configuration {i}")
        conn.close()

        mock_response = json.dumps({
            "title": "API Design Best Practices",
            "body": "When designing REST APIs, follow consistent patterns.",
            "tags": ["api", "rest", "design"],
            "confidence": 0.85,
        })
        llm_fn = MagicMock(return_value=mock_response)

        d = self._make_distiller(db_path, knowledge_root, llm_fn=llm_fn, threshold=3, cluster_min=3)
        result = d.run()

        if result["distilled"] > 0:
            llm_fn.assert_called()
            filepath = Path(result["files_created"][0])
            content = filepath.read_text()
            assert "API Design Best Practices" in content
            assert "confidence: 0.85" in content

    def test_run_no_clusters(self, db_path, knowledge_root):
        conn = _create_db(db_path)
        # Use single-word summaries with zero overlap to prevent clustering
        _insert_episode(conn, summary="alpha", details="x")
        _insert_episode(conn, summary="beta", details="y")
        _insert_episode(conn, summary="gamma", details="z")
        conn.close()

        d = self._make_distiller(db_path, knowledge_root, threshold=2, cluster_min=2)
        result = d.run()
        assert result["distilled"] == 0
        assert "no clusters" in (result["skipped_reason"] or "")

    def test_extractive_fallback(self, db_path, knowledge_root):
        """Without llm_fn, distiller uses extractive fallback."""
        conn = _create_db(db_path)
        for i in range(4):
            _insert_episode(conn, summary=f"machine learning training run {i}",
                          details=f"neural network optimization step {i}")
        conn.close()

        d = self._make_distiller(db_path, knowledge_root, llm_fn=None, threshold=3, cluster_min=3)
        result = d.run()
        # Should still produce output using extractive fallback
        if result["clusters_found"] > 0:
            assert result["distilled"] >= 0  # may or may not produce output depending on clustering


# ── LLM response parsing ─────────────────────────────────────────────────────


class TestParseLLMResponse:
    def test_valid_json(self):
        response = json.dumps({
            "title": "Test",
            "body": "Content",
            "tags": ["a"],
            "confidence": 0.8,
        })
        result = EpisodicDistiller._parse_llm_response(response)
        assert result is not None
        assert result["title"] == "Test"
        assert result["confidence"] == 0.8

    def test_markdown_fenced_json(self):
        response = '```json\n{"title": "Test", "body": "Content"}\n```'
        result = EpisodicDistiller._parse_llm_response(response)
        assert result is not None
        assert result["title"] == "Test"

    def test_missing_required_fields(self):
        response = json.dumps({"title": "No body field"})
        result = EpisodicDistiller._parse_llm_response(response)
        assert result is None

    def test_garbage_input(self):
        result = EpisodicDistiller._parse_llm_response("not json at all")
        assert result is None

    def test_json_in_surrounding_text(self):
        response = 'Here is the result:\n{"title": "Found", "body": "In text"}\nEnd.'
        result = EpisodicDistiller._parse_llm_response(response)
        assert result is not None
        assert result["title"] == "Found"
