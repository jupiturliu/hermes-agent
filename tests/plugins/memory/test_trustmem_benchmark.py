"""Latency and throughput benchmarks for TrustMemMemoryProvider."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from plugins.memory.trustmem import TrustMemMemoryProvider

_SEARCH_LATENCY_BUDGET_MS = 50   # mock search must complete in <50ms
_PREFETCH_BUDGET_MS = 100        # prefetch (sync path) must complete in <100ms


def _make_provider(tmp_dir: str) -> TrustMemMemoryProvider:
    root = Path(tmp_dir) / "trustmem"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "knowledge_search.py").write_text("# stub")
    os.environ["TRUSTMEM_ROOT"] = str(root)
    p = TrustMemMemoryProvider()
    p.initialize("bench-session")
    return p


def _install_mock_ks(provider: TrustMemMemoryProvider, n_results: int = 5) -> MagicMock:
    mock_ks = MagicMock()
    mock_ks.search.return_value = [
        {"title": f"Doc {i}", "effective_confidence": 0.8, "snippet": "x" * 100,
         "verification_status": "peer_reviewed", "score": 10 - i, "domain": "test"}
        for i in range(n_results)
    ]
    provider._ks = mock_ks
    return mock_ks


class SearchLatencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = _make_provider(self.tmp.name)
        _install_mock_ks(self.provider)

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)

    def test_tool_search_latency_under_budget(self):
        """tool_search with mock backend must complete in <50ms."""
        t0 = time.perf_counter()
        result = self.provider.handle_tool_call("trustmem_search", {"query": "benchmark", "top": 5})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed_ms, _SEARCH_LATENCY_BUDGET_MS,
                        f"search took {elapsed_ms:.1f}ms, budget {_SEARCH_LATENCY_BUDGET_MS}ms")
        data = json.loads(result)
        self.assertIn("results", data)

    def test_tool_search_throughput_10_queries(self):
        """10 sequential mock searches must complete in <500ms total."""
        t0 = time.perf_counter()
        for _ in range(10):
            self.provider.handle_tool_call("trustmem_search", {"query": "test", "top": 3})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed_ms, 500, f"10 queries took {elapsed_ms:.1f}ms")


class PrefetchLatencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = _make_provider(self.tmp.name)
        _install_mock_ks(self.provider)

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)

    def test_prefetch_sync_path_under_budget(self):
        """First-turn synchronous prefetch must complete in <100ms with mock backend."""
        t0 = time.perf_counter()
        ctx = self.provider.prefetch("transformer attention")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed_ms, _PREFETCH_BUDGET_MS,
                        f"prefetch took {elapsed_ms:.1f}ms, budget {_PREFETCH_BUDGET_MS}ms")
        self.assertIn("Recalled Context", ctx)

    def test_queue_prefetch_non_blocking(self):
        """queue_prefetch must return immediately (<5ms) without blocking."""
        t0 = time.perf_counter()
        self.provider.queue_prefetch("background query")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed_ms, 5, f"queue_prefetch blocked for {elapsed_ms:.1f}ms")
        # Wait for background thread
        if self.provider._prefetch_thread:
            self.provider._prefetch_thread.join(timeout=3)


class ResultFormatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = _make_provider(self.tmp.name)
        _install_mock_ks(self.provider, n_results=3)

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)

    def test_search_result_has_title_and_confidence(self):
        """Each result in trustmem_search output has title and effective_confidence."""
        raw = self.provider.handle_tool_call("trustmem_search", {"query": "test", "top": 3})
        data = json.loads(raw)
        for r in data["results"]:
            self.assertIn("title", r)
            self.assertIn("effective_confidence", r)

    def test_prefetch_returns_markdown_context(self):
        """prefetch result starts with ## Recalled Context header."""
        ctx = self.provider.prefetch("test query")
        self.assertTrue(ctx.startswith("## Recalled Context"))

    def test_search_respects_top_parameter(self):
        """top=2 returns at most 2 results."""
        _install_mock_ks(self.provider, n_results=5)
        raw = self.provider.handle_tool_call("trustmem_search", {"query": "test", "top": 2})
        data = json.loads(raw)
        self.assertLessEqual(len(data["results"]), 2)


class ConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = _make_provider(self.tmp.name)
        _install_mock_ks(self.provider)

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)

    def test_concurrent_sync_turns_no_deadlock(self):
        """Multiple concurrent sync_turn calls must not deadlock."""
        import threading
        mock_el = MagicMock()
        self.provider._el = mock_el
        threads = []
        for i in range(5):
            t = threading.Thread(
                target=self.provider.sync_turn,
                args=(f"user msg {i}", f"assistant msg {i}"),
            )
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        # All threads should have finished
        self.assertTrue(all(not t.is_alive() for t in threads))


if __name__ == "__main__":
    unittest.main()
