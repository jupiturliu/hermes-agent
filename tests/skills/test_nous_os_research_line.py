"""Structural tests for the NOUS OS Research Line skills.

These tests do not invoke Hermes; they validate the SKILL.md files in
`skills/nous-os/` are well-formed and keep the load-bearing contract
clauses from the Wave 4 spec
(`docs/research-line/hermes-integration.md` in jupiturliu/nous-os).

Written as stdlib unittest so they run anywhere — including environments
where the repo's pytest conftest is incompatible with the host Python.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOUS_OS_BUCKET = REPO_ROOT / "skills" / "nous-os"
L2_SKILL = NOUS_OS_BUCKET / "research-line-l2-triage" / "SKILL.md"
L3_SKILL = NOUS_OS_BUCKET / "research-line-l3-synthesis" / "SKILL.md"


def _frontmatter(skill_path: Path) -> dict:
    """Extract minimal YAML-like frontmatter from a SKILL.md. Stdlib only."""

    text = skill_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    block = text[4:end]
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line and not line.startswith(" "):
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip().strip('"')
    return fields


class BucketExistsTests(unittest.TestCase):
    def test_nous_os_bucket_has_description(self) -> None:
        self.assertTrue(
            (NOUS_OS_BUCKET / "DESCRIPTION.md").exists(),
            "skills/nous-os/DESCRIPTION.md must exist so the bucket is indexed",
        )

    def test_both_skill_files_exist(self) -> None:
        self.assertTrue(L2_SKILL.exists(), "L2 SKILL.md must exist")
        self.assertTrue(L3_SKILL.exists(), "L3 SKILL.md must exist")


class L2FrontmatterTests(unittest.TestCase):
    def test_name_and_version(self) -> None:
        fm = _frontmatter(L2_SKILL)
        self.assertEqual(fm.get("name"), "research-line-l2-triage")
        self.assertIn("version", fm)

    def test_description_mentions_weekly_and_operator(self) -> None:
        fm = _frontmatter(L2_SKILL)
        description = fm.get("description", "").lower()
        self.assertIn("weekly", description)
        self.assertTrue(
            "never auto-merge" in description or "operator" in description,
            "L2 description must signal operator-gating",
        )


class L3FrontmatterTests(unittest.TestCase):
    def test_name_and_version(self) -> None:
        fm = _frontmatter(L3_SKILL)
        self.assertEqual(fm.get("name"), "research-line-l3-synthesis")
        self.assertIn("version", fm)

    def test_description_mentions_bi_weekly_and_operator(self) -> None:
        fm = _frontmatter(L3_SKILL)
        description = fm.get("description", "").lower()
        self.assertIn("bi-weekly", description)
        self.assertTrue(
            "never auto-merge" in description or "operator" in description,
            "L3 description must signal operator-gating",
        )


class L2ContractClauseTests(unittest.TestCase):
    """The L2 SKILL.md must reproduce the load-bearing contract clauses from
    Wave 4 spec in jupiturliu/nous-os."""

    def setUp(self) -> None:
        self.body = L2_SKILL.read_text(encoding="utf-8")

    def test_required_clauses_present(self) -> None:
        required = (
            "Never auto-merge",
            "Never edit existing inbound notes",
            "Never modify structural sections",
            "Never make HTTP requests",
            "Never refer to the operator",
            "Cap at 3 promotions per week",
            "research-line:l2-triage",
            "docs/research-line/hermes-integration.md",
            "docs/research-line/inbound/_template.md",
            "anchor-atlas",
        )
        missing = [c for c in required if c not in self.body]
        self.assertEqual(missing, [], f"L2 skill missing required clauses: {missing!r}")


class L3ContractClauseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.body = L3_SKILL.read_text(encoding="utf-8")

    def test_required_clauses_present(self) -> None:
        required = (
            "Never auto-merge",
            "Never edit existing synthesis files",
            "Never make causal claims that exceed",
            "Never use",
            "Never extrapolate Sandbox findings to trading-agent",
            "Never recommend dropping a method commitment",
            "research-line:l3-synthesis",
            "docs/research-line/synthesis/_template.md",
            "every Sunday at 14:00 UTC",
            "13-day",
            "direction signal, not validation",
        )
        missing = [c for c in required if c not in self.body]
        self.assertEqual(missing, [], f"L3 skill missing required clauses: {missing!r}")

    def test_required_synthesis_section_anchors_present(self) -> None:
        sections = (
            "Period at a glance",
            "Most influential inbound notes",
            "Coverage observations",
            "Instrument signals",
            "What we were wrong about",
            "Next period's planned shifts",
        )
        missing = [s for s in sections if s not in self.body]
        self.assertEqual(missing, [], f"L3 skill missing section anchors: {missing!r}")


class CrossReferenceTests(unittest.TestCase):
    def test_l3_references_l2_as_related(self) -> None:
        body = L3_SKILL.read_text(encoding="utf-8")
        self.assertIn(
            "research-line-l2-triage",
            body,
            "L3 skill should reference L2 as a related skill",
        )


if __name__ == "__main__":
    unittest.main()
