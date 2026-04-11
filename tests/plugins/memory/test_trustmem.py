"""Unit tests for TrustMemMemoryProvider."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure plugin is importable
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from plugins.memory.trustmem import TrustMemMemoryProvider


def _make_provider(tmp_dir: str, root_exists: bool = True) -> TrustMemMemoryProvider:
    p = TrustMemMemoryProvider()
    if root_exists:
        # Create fake trustmem root with required sentinel file
        root = Path(tmp_dir) / "trustmem"
        (root / "tools").mkdir(parents=True)
        (root / "tools" / "knowledge_search.py").write_text("# stub")
        os.environ["TRUSTMEM_ROOT"] = str(root)
    else:
        os.environ.pop("TRUSTMEM_ROOT", None)
    return p


class AvailabilityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)

    def test_available_when_root_and_sentinel_exist(self):
        p = _make_provider(self.tmp.name, root_exists=True)
        self.assertTrue(p.is_available())

    def test_unavailable_when_root_missing(self):
        os.environ.pop("TRUSTMEM_ROOT", None)
        p = TrustMemMemoryProvider()
        # Override auto-detect candidates to non-existent paths
        with patch.object(p, "_resolve_root", return_value=None):
            self.assertFalse(p.is_available())


class InitializeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = _make_provider(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)

    def test_initialize_sets_session_id(self):
        self.provider.initialize("sess-001")
        self.assertEqual(self.provider._session_id, "sess-001")

    def test_initialize_defaults_agent_name(self):
        os.environ.pop("TRUSTMEM_AGENT", None)
        self.provider.initialize("sess-002")
        self.assertEqual(self.provider._agent_name, "hermes")

    def test_initialize_reads_agent_name_from_env(self):
        os.environ["TRUSTMEM_AGENT"] = "aria"
        self.provider.initialize("sess-003")
        self.assertEqual(self.provider._agent_name, "aria")
        os.environ.pop("TRUSTMEM_AGENT")

    def test_cron_context_sets_readonly(self):
        self.provider.initialize("sess-004", agent_context="cron")
        self.assertTrue(self.provider._readonly)

    def test_primary_context_not_readonly(self):
        self.provider.initialize("sess-005", agent_context="primary")
        self.assertFalse(self.provider._readonly)


class SystemPromptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = _make_provider(self.tmp.name)
        self.provider.initialize("sess-sp")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)

    def test_system_prompt_contains_trustmem_heading(self):
        block = self.provider.system_prompt_block()
        self.assertIn("TrustMem", block)

    def test_system_prompt_empty_in_readonly_mode(self):
        self.provider._readonly = True
        self.assertEqual(self.provider.system_prompt_block(), "")


class ToolSchemasTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = _make_provider(self.tmp.name)
        self.provider.initialize("sess-tools")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)

    def test_returns_four_tool_schemas(self):
        schemas = self.provider.get_tool_schemas()
        self.assertEqual(len(schemas), 4)
        names = {s["name"] for s in schemas}
        self.assertIn("trustmem_search", names)
        self.assertIn("trustmem_reason", names)
        self.assertIn("trustmem_distill", names)
        self.assertIn("trustmem_stats", names)

    def test_search_schema_has_required_query(self):
        schema = next(s for s in self.provider.get_tool_schemas() if s["name"] == "trustmem_search")
        self.assertIn("query", schema["parameters"]["required"])

    def test_reason_schema_has_required_question(self):
        schema = next(s for s in self.provider.get_tool_schemas() if s["name"] == "trustmem_reason")
        self.assertIn("question", schema["parameters"]["required"])


class ToolCallTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = _make_provider(self.tmp.name)
        self.provider.initialize("sess-tools")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)

    def test_search_returns_error_when_ks_none(self):
        self.provider._ks = None
        result = json.loads(self.provider.handle_tool_call("trustmem_search", {"query": "test"}))
        self.assertIn("error", result)

    def test_search_calls_ks_search(self):
        mock_ks = MagicMock()
        mock_ks.search.return_value = [{"title": "Doc A", "effective_confidence": 0.9, "snippet": "..."}]
        self.provider._ks = mock_ks
        result = json.loads(self.provider.handle_tool_call("trustmem_search", {"query": "foo", "top": 3}))
        self.assertIn("results", result)
        mock_ks.search.assert_called_once()

    def test_unknown_tool_returns_error(self):
        result = json.loads(self.provider.handle_tool_call("nonexistent_tool", {}))
        self.assertIn("error", result)


class ConfigSchemaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = _make_provider(self.tmp.name)
        self.provider.initialize("sess-cfg")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)

    def test_config_schema_has_trustmem_root(self):
        schema = self.provider.get_config_schema()
        keys = {s["key"] for s in schema}
        self.assertIn("trustmem_root", keys)

    def test_trustmem_root_is_required(self):
        schema = self.provider.get_config_schema()
        root_cfg = next(s for s in schema if s["key"] == "trustmem_root")
        self.assertTrue(root_cfg["required"])

    def test_save_config_writes_json(self):
        hermes_home = self.tmp.name
        self.provider.save_config({"trustmem_root": "/tmp/trustmem"}, hermes_home)
        cfg_file = Path(hermes_home) / "trustmem.json"
        self.assertTrue(cfg_file.exists())
        data = json.loads(cfg_file.read_text())
        self.assertEqual(data["trustmem_root"], "/tmp/trustmem")

    def test_save_config_merges_existing(self):
        hermes_home = self.tmp.name
        cfg_file = Path(hermes_home) / "trustmem.json"
        cfg_file.write_text(json.dumps({"existing_key": "value"}))
        self.provider.save_config({"trustmem_root": "/tmp/t"}, hermes_home)
        data = json.loads(cfg_file.read_text())
        self.assertIn("existing_key", data)
        self.assertIn("trustmem_root", data)


class SyncTurnTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = _make_provider(self.tmp.name)
        self.provider.initialize("sess-sync")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)

    def test_sync_turn_noop_when_readonly(self):
        self.provider._readonly = True
        self.provider.sync_turn("hello", "world")
        self.assertIsNone(self.provider._sync_thread)

    def test_sync_turn_noop_when_empty_user_content(self):
        self.provider.sync_turn("  ", "response")
        self.assertIsNone(self.provider._sync_thread)

    def test_sync_turn_spawns_thread(self):
        mock_el = MagicMock()
        self.provider._el = mock_el
        self.provider._quality_threshold = 0.0  # accept everything for this test
        self.provider.sync_turn("user message", "assistant response")
        self.assertIsNotNone(self.provider._sync_thread)
        self.provider._sync_thread.join(timeout=3)
        mock_el.log_episode.assert_called_once()

    def test_sync_turn_filters_trivial(self):
        mock_el = MagicMock()
        self.provider._el = mock_el
        self.provider._quality_threshold = 0.3
        self.provider.sync_turn("ok", "sure")
        if self.provider._sync_thread:
            self.provider._sync_thread.join(timeout=3)
        mock_el.log_episode.assert_not_called()


class ShutdownTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = _make_provider(self.tmp.name)
        self.provider.initialize("sess-shutdown")

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TRUSTMEM_ROOT", None)

    def test_shutdown_joins_threads(self):
        done = threading.Event()

        def slow():
            done.wait(timeout=2)

        t = threading.Thread(target=slow, daemon=True)
        t.start()
        self.provider._sync_thread = t
        done.set()
        self.provider.shutdown()
        self.assertFalse(t.is_alive())


if __name__ == "__main__":
    unittest.main()
