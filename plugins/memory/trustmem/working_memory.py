"""
TrustMem Working Memory Assembler
===================================
Implements Databricks-inspired structured two-layer retrieval for prompt
injection. At each turn, working memory is dynamically assembled from:

  1. **Semantic layer** (high priority) — distilled knowledge rules, verified
     research, high-confidence entries. These provide generalizable guidance.
  2. **Episodic layer** (supporting) — specific past task outcomes and
     observations. These provide concrete examples and precedents.

The assembler retrieves from both layers, deduplicates, and formats them
into a structured markdown block that the LLM can parse hierarchically:
  - Semantic rules go first (the agent should follow these)
  - Episodic examples follow (the agent can reference these)

This replaces the flat _do_search approach where all results were mixed
together with no priority distinction.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_SEMANTIC_SLOTS = 3   # max semantic entries in working memory
DEFAULT_EPISODIC_SLOTS = 3   # max episodic entries in working memory
DEFAULT_MAX_SNIPPET = 300    # max chars per snippet

# Knowledge entries with these verification statuses are treated as semantic
_SEMANTIC_STATUSES = {"peer_reviewed", "auto_distilled", "verified", "expert_reviewed"}

# Minimum effective_confidence to qualify as semantic (otherwise demoted to episodic)
_SEMANTIC_CONFIDENCE_FLOOR = 0.4


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single entry in working memory, tagged with its layer."""
    title: str
    snippet: str
    score: float
    confidence: float
    layer: str               # "semantic" or "episodic"
    source: str = ""         # file path or episode id
    verification: str = ""   # verification_status
    domain: str = ""
    age_days: int = 0


@dataclass
class WorkingMemory:
    """Assembled working memory for one turn."""
    semantic: list[MemoryEntry] = field(default_factory=list)
    episodic: list[MemoryEntry] = field(default_factory=list)
    query: str = ""

    @property
    def total_entries(self) -> int:
        return len(self.semantic) + len(self.episodic)

    @property
    def is_empty(self) -> bool:
        return self.total_entries == 0


# ── Classification ───────────────────────────────────────────────────────────

def classify_result(result: dict[str, Any]) -> str:
    """
    Classify a knowledge_search result as 'semantic' or 'episodic'.

    Semantic (generalizable rules/knowledge):
      - Verified or distilled entries with high confidence
      - Entries from knowledge domains with peer review

    Episodic (specific instances):
      - Unverified or low-confidence entries
      - Raw conversation logs or task records
    """
    verification = result.get("verification_status", "")
    confidence = result.get("effective_confidence", 0)
    memory_type = result.get("memory_type", "")

    # Explicit memory_type overrides
    if memory_type == "semantic":
        return "semantic"
    if memory_type == "episodic":
        return "episodic"

    # Classification by verification + confidence
    if verification in _SEMANTIC_STATUSES and confidence >= _SEMANTIC_CONFIDENCE_FLOOR:
        return "semantic"

    return "episodic"


def classify_episode(episode: dict[str, Any]) -> MemoryEntry:
    """Convert a recall_similar episode tuple into a MemoryEntry."""
    task = episode.get("task", "")
    approach = episode.get("approach", "")
    notes = episode.get("notes", "")
    quality = episode.get("quality_score", 0.5)

    snippet = task
    if notes:
        snippet += f" — {notes}"

    return MemoryEntry(
        title=task[:100],
        snippet=snippet[:DEFAULT_MAX_SNIPPET],
        score=quality,
        confidence=quality,
        layer="episodic",
        source=episode.get("id", ""),
        domain=episode.get("agent", ""),
    )


# ── Assembly ─────────────────────────────────────────────────────────────────

def assemble_working_memory(
    query: str,
    knowledge_results: list[dict[str, Any]] | None = None,
    episode_results: list[tuple[dict, float]] | None = None,
    semantic_slots: int = DEFAULT_SEMANTIC_SLOTS,
    episodic_slots: int = DEFAULT_EPISODIC_SLOTS,
) -> WorkingMemory:
    """
    Assemble working memory from knowledge search results and episode recalls.

    Args:
        query: The current user query
        knowledge_results: Results from knowledge_search.search()
        episode_results: Results from episode_logger.recall_similar()
            Each is (episode_dict, similarity_score)
        semantic_slots: Max semantic entries to include
        episodic_slots: Max episodic entries to include

    Returns:
        WorkingMemory with classified, deduplicated, priority-ordered entries.
    """
    wm = WorkingMemory(query=query)
    seen_titles: set[str] = set()

    # Phase 1: Classify knowledge results into semantic vs episodic
    if knowledge_results:
        for r in knowledge_results:
            layer = classify_result(r)
            title = r.get("title") or r.get("file", "untitled")

            # Dedup by title
            title_key = title.lower().strip()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            entry = MemoryEntry(
                title=title,
                snippet=(r.get("snippet") or r.get("body", ""))[:DEFAULT_MAX_SNIPPET],
                score=r.get("score", 0),
                confidence=r.get("effective_confidence", 0),
                layer=layer,
                source=r.get("file", ""),
                verification=r.get("verification_status", ""),
                domain=r.get("domain", ""),
                age_days=r.get("age_days", 0),
            )

            if layer == "semantic":
                wm.semantic.append(entry)
            else:
                wm.episodic.append(entry)

    # Phase 2: Add episode recalls (always episodic)
    if episode_results:
        for ep_tuple in episode_results:
            ep, sim = ep_tuple if isinstance(ep_tuple, tuple) else (ep_tuple, 0.5)
            title_key = (ep.get("task", "") or "")[:100].lower().strip()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            entry = classify_episode(ep)
            entry.score = sim if isinstance(sim, (int, float)) else entry.score
            wm.episodic.append(entry)

    # Phase 3: Sort by score within each layer, apply slot limits
    wm.semantic.sort(key=lambda e: e.score, reverse=True)
    wm.episodic.sort(key=lambda e: e.score, reverse=True)
    wm.semantic = wm.semantic[:semantic_slots]
    wm.episodic = wm.episodic[:episodic_slots]

    return wm


# ── Formatting ───────────────────────────────────────────────────────────────

def _format_entry(entry: MemoryEntry, idx: int) -> str:
    """Format a single memory entry as markdown."""
    confidence_str = f"{entry.confidence:.2f}" if entry.confidence else "?"
    header = f"{idx}. **{entry.title}**"
    if entry.verification:
        header += f" [{entry.verification}]"
    header += f" (confidence: {confidence_str})"
    return f"{header}\n   {entry.snippet}"


def format_working_memory(wm: WorkingMemory) -> str:
    """
    Format assembled working memory as structured markdown for prompt injection.

    Structure:
      ## Recalled Context
      ### Guidelines (semantic knowledge)
      1. ...
      ### Relevant History (episodic memory)
      1. ...
    """
    if wm.is_empty:
        return ""

    sections: list[str] = ["## Recalled Context"]

    if wm.semantic:
        sections.append("### Guidelines")
        sections.append("_Verified rules and patterns — follow these when applicable._")
        for i, entry in enumerate(wm.semantic, 1):
            sections.append(_format_entry(entry, i))

    if wm.episodic:
        sections.append("### Relevant History")
        sections.append("_Past observations and task outcomes — use as reference._")
        for i, entry in enumerate(wm.episodic, 1):
            sections.append(_format_entry(entry, i))

    return "\n\n".join(sections)
