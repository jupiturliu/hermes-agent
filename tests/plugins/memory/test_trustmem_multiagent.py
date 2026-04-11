"""Multi-agent scenario tests for TrustMemMemoryProvider."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from plugins.memory.trustmem import TrustMemMemoryProvider


def _make_provider(tmp_dir: str, agent: str = "hermes") -> TrustMemMemoryProvider:
    root = Path(tmp_dir) / "trustmem"
    (root / "tools").mkdir(parents=True, exist_ok=True)
    (root / "tools" / "knowledge_search.py").write_text("# stub")
    os.environ["TRUSTMEM_ROOT"] = str(root)
    os.environ["TRUSTMEM_AGENT"] = agent
    p = TrustMemMemoryProvider()
    p.initialize(f"session-{agent}")
    p._quality_threshold = 0.0  # bypass quality filter for unit tests
    return p


class DelegationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.parent = _make_provider(self.tmp.name, agent="aria")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)
        os.environ.pop("TRUSTMEM_AGENT", None)

    def test_on_delegation_logs_episode(self):
        """on_delegation logs subagent result as episode with task_type=delegation."""
        mock_el = MagicMock()
        self.parent._el = mock_el
        self.parent.on_delegation(
            task="Analyze HBM4 market",
            result="HBM4 launches in Q3 2026",
            child_session_id="research-001",
        )
        # Wait for daemon thread
        import time; time.sleep(0.1)
        mock_el.log_episode.assert_called_once()
        kwargs = mock_el.log_episode.call_args[1]
        self.assertEqual(kwargs["task_type"], "delegation")
        self.assertIn("research-001", kwargs["approach"])

    def test_on_delegation_noop_when_readonly(self):
        """on_delegation is silent when provider is in readonly mode."""
        mock_el = MagicMock()
        self.parent._el = mock_el
        self.parent._readonly = True
        self.parent.on_delegation("task", "result")
        import time; time.sleep(0.05)
        mock_el.log_episode.assert_not_called()


class MemoryWriteMirrorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = _make_provider(self.tmp.name, agent="aria")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)
        os.environ.pop("TRUSTMEM_AGENT", None)

    def test_on_memory_write_logs_episode(self):
        """on_memory_write creates episode with task_type=memory-write."""
        mock_el = MagicMock()
        self.provider._el = mock_el
        self.provider.on_memory_write(
            action="update",
            target="MEMORY.md",
            content="User prefers concise responses.",
        )
        import time; time.sleep(0.1)
        mock_el.log_episode.assert_called_once()
        kwargs = mock_el.log_episode.call_args[1]
        self.assertEqual(kwargs["task_type"], "memory-write")
        self.assertIn("update", kwargs["task"])

    def test_on_memory_write_noop_when_readonly(self):
        mock_el = MagicMock()
        self.provider._el = mock_el
        self.provider._readonly = True
        self.provider.on_memory_write("write", "MEMORY.md", "content")
        import time; time.sleep(0.05)
        mock_el.log_episode.assert_not_called()


class SessionEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = _make_provider(self.tmp.name, agent="hermes")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)
        os.environ.pop("TRUSTMEM_AGENT", None)

    def test_on_session_end_logs_summary(self):
        """on_session_end logs a session-summary episode."""
        mock_el = MagicMock()
        self.provider._el = mock_el
        messages = [
            {"role": "user", "content": "What is CXL?"},
            {"role": "assistant", "content": "CXL is a cache-coherent interconnect..."},
            {"role": "user", "content": "How does it compare to PCIe?"},
        ]
        self.provider.on_session_end(messages)
        mock_el.log_episode.assert_called_once()
        kwargs = mock_el.log_episode.call_args[1]
        self.assertEqual(kwargs["task_type"], "session-summary")

    def test_on_session_end_noop_with_no_user_turns(self):
        mock_el = MagicMock()
        self.provider._el = mock_el
        messages = [{"role": "assistant", "content": "Hello"}]
        self.provider.on_session_end(messages)
        mock_el.log_episode.assert_not_called()


class PreCompressTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = _make_provider(self.tmp.name, agent="hermes")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)
        os.environ.pop("TRUSTMEM_AGENT", None)

    def test_on_pre_compress_logs_episode(self):
        """on_pre_compress logs last 3 user turns before compression."""
        mock_el = MagicMock()
        self.provider._el = mock_el
        messages = [
            {"role": "user", "content": f"Turn {i}"}
            for i in range(5)
        ]
        result = self.provider.on_pre_compress(messages)
        self.assertEqual(result, "")  # must not inject text
        mock_el.log_episode.assert_called_once()
        kwargs = mock_el.log_episode.call_args[1]
        self.assertEqual(kwargs["task_type"], "pre-compress")

    def test_on_pre_compress_returns_empty_string(self):
        """Return value must always be empty string (not injected into context)."""
        mock_el = MagicMock()
        self.provider._el = mock_el
        result = self.provider.on_pre_compress([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "")


class MultiAgentIsolationTest(unittest.TestCase):
    """Verify two providers with different agent names produce separate attributions."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)
        os.environ.pop("TRUSTMEM_AGENT", None)

    def test_two_agents_log_separate_episodes(self):
        aria = _make_provider(self.tmp.name, agent="aria")
        research = _make_provider(self.tmp.name, agent="research")

        aria_el = MagicMock()
        research_el = MagicMock()
        aria._el = aria_el
        research._el = research_el

        aria.sync_turn("aria query", "aria answer")
        research.sync_turn("research query", "research answer")

        import time; time.sleep(0.15)

        aria_el.log_episode.assert_called_once()
        research_el.log_episode.assert_called_once()

        aria_kwargs = aria_el.log_episode.call_args[1]
        research_kwargs = research_el.log_episode.call_args[1]

        self.assertEqual(aria_kwargs["agent"], "aria")
        self.assertEqual(research_kwargs["agent"], "research")


if __name__ == "__main__":
    unittest.main()
