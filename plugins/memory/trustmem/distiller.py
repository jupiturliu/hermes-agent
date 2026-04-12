"""
TrustMem Episodic → Semantic Distillation Engine
=================================================
Periodically consolidates raw episodic memories into generalizable
semantic knowledge entries, inspired by Databricks MemAlign approach:

  1. Fetch unconsolidated ("raw") episodes from SQLite
  2. Cluster by topic similarity
  3. For each cluster, distill generalizable rules/patterns via LLM
  4. Write results as knowledge markdown files with frontmatter
  5. Mark source episodes as "consolidated"

The distiller can run:
  - Automatically at session end (when raw episode count exceeds threshold)
  - Manually via the `trustmem_distill` agent tool
  - As a standalone CLI: `python -m plugins.memory.trustmem.distiller`

Configuration (env vars or passed at init):
  TRUSTMEM_DISTILL_THRESHOLD  — min raw episodes to trigger (default: 10)
  TRUSTMEM_DISTILL_MODEL      — LLM model for summarization (default: use agent's model)
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 10          # min raw episodes before distillation triggers
DEFAULT_CLUSTER_MIN = 3         # min episodes in a cluster to distill
DEFAULT_MAX_EPISODES = 100      # max episodes per distillation run
DEFAULT_KNOWLEDGE_DOMAIN = "distilled"  # subdirectory under knowledge/


# ── Episode fetching ─────────────────────────────────────────────────────────

def fetch_raw_episodes(
    db_path: str,
    limit: int = DEFAULT_MAX_EPISODES,
    agent_id: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch unconsolidated episodes from SQLite, ordered by recency."""
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        query = (
            "SELECT id, agent_id, created_at, summary, details, importance, "
            "topic_tags, consolidation_status, consolidation_count "
            "FROM episodes WHERE consolidation_status = 'raw'"
        )
        params: list[Any] = []
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.debug("distiller: fetch error: %s", exc)
        return []


def count_raw_episodes(db_path: str, agent_id: str | None = None) -> int:
    """Quick count of unconsolidated episodes."""
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        query = "SELECT COUNT(*) FROM episodes WHERE consolidation_status = 'raw'"
        params: list[Any] = []
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        count = conn.execute(query, params).fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def mark_episodes_consolidated(
    db_path: str,
    episode_ids: list[str],
    status: str = "consolidated",
) -> int:
    """Update consolidation_status and increment consolidation_count."""
    if not episode_ids:
        return 0
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        placeholders = ",".join("?" for _ in episode_ids)
        conn.execute(
            f"UPDATE episodes SET consolidation_status = ?, "
            f"consolidation_count = consolidation_count + 1 "
            f"WHERE id IN ({placeholders})",
            [status, *episode_ids],
        )
        conn.commit()
        updated = conn.total_changes
        conn.close()
        return updated
    except Exception as exc:
        logger.debug("distiller: mark error: %s", exc)
        return 0


# ── Clustering ───────────────────────────────────────────────────────────────

def _extract_keywords(text: str) -> set[str]:
    """Simple keyword extraction for clustering (CJK-aware)."""
    text = text.lower()
    # Extract CJK sequences and Latin words
    tokens = re.findall(r'[\u4e00-\u9fff]+|[a-z][a-z0-9_-]{1,}', text)
    # Split long CJK sequences into bigrams
    expanded = []
    for t in tokens:
        if re.match(r'[\u4e00-\u9fff]', t) and len(t) > 2:
            expanded.extend(t[i:i+2] for i in range(len(t) - 1))
        else:
            expanded.append(t)
    return set(expanded)


def cluster_episodes(
    episodes: list[dict[str, Any]],
    min_cluster_size: int = DEFAULT_CLUSTER_MIN,
) -> list[list[dict[str, Any]]]:
    """
    Group episodes by topic similarity using keyword overlap.
    Returns clusters of size >= min_cluster_size.
    Unclustered episodes with tag overlap are also grouped.
    """
    # Extract keywords per episode
    ep_keywords: list[tuple[dict, set[str]]] = []
    for ep in episodes:
        text = f"{ep.get('summary', '')} {ep.get('details', '')}"
        tags = ep.get("topic_tags")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []
        kw = _extract_keywords(text)
        if tags:
            kw.update(t.lower() for t in tags if isinstance(t, str))
        ep_keywords.append((ep, kw))

    n = len(ep_keywords)
    assigned = [False] * n
    clusters: list[list[dict[str, Any]]] = []

    # Greedy clustering: for each unassigned episode, gather similar ones
    for i in range(n):
        if assigned[i]:
            continue
        cluster = [ep_keywords[i][0]]
        assigned[i] = True
        kw_i = ep_keywords[i][1]

        if not kw_i:
            continue

        for j in range(i + 1, n):
            if assigned[j]:
                continue
            kw_j = ep_keywords[j][1]
            if not kw_j:
                continue
            # Jaccard similarity
            intersection = len(kw_i & kw_j)
            union = len(kw_i | kw_j)
            if union > 0 and intersection / union >= 0.15:
                cluster.append(ep_keywords[j][0])
                assigned[j] = True
                # Expand cluster keywords
                kw_i = kw_i | kw_j

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    return clusters


# ── Distillation prompt ──────────────────────────────────────────────────────

def build_distillation_prompt(cluster: list[dict[str, Any]]) -> str:
    """Build a prompt that asks the LLM to distill episodes into a semantic rule."""
    episode_texts = []
    for i, ep in enumerate(cluster, 1):
        summary = ep.get("summary", "")
        details = ep.get("details", "")
        date = ep.get("created_at", "unknown")
        importance = ep.get("importance", 0.5)
        episode_texts.append(
            f"Episode {i} [{date}] (importance: {importance}):\n"
            f"  Summary: {summary}\n"
            f"  Details: {details[:300]}"
        )

    episodes_block = "\n\n".join(episode_texts)

    return f"""You are a memory consolidation engine. Analyze these {len(cluster)} episodic memories from an AI agent's work history and distill them into ONE generalizable semantic rule or knowledge entry.

## Episodes

{episodes_block}

## Instructions

1. Identify the common theme, pattern, or lesson across these episodes.
2. Write a concise, generalizable rule or knowledge entry (not episode-specific).
3. Include concrete guidance that would help the agent in future similar situations.
4. Rate your confidence in this distillation (0.0 to 1.0).

## Output Format (JSON)

{{
  "title": "Short descriptive title (English or Chinese, match the episodes' language)",
  "body": "The distilled knowledge entry. 2-4 paragraphs covering: what the pattern is, why it matters, and how to apply it.",
  "tags": ["tag1", "tag2", "tag3"],
  "confidence": 0.7,
  "language": "en or zh"
}}

Return ONLY the JSON object, no markdown fences."""


# ── Knowledge file writing ───────────────────────────────────────────────────

def write_knowledge_file(
    knowledge_root: Path,
    title: str,
    body: str,
    tags: list[str],
    confidence: float,
    source_episode_ids: list[str],
    agent: str = "hermes",
    domain: str = DEFAULT_KNOWLEDGE_DOMAIN,
) -> Path:
    """Write a distilled knowledge entry as a markdown file with frontmatter.

    Uses knowledge_utils.write_frontmatter when available (proper YAML formatting),
    falls back to manual writing if TrustMem tools are not importable.
    """
    domain_dir = knowledge_root / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from title
    slug = re.sub(r'[^\w\s-]', '', title.lower())
    slug = re.sub(r'[\s]+', '-', slug).strip('-')[:60]
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{slug}-{date_str}.md"
    filepath = domain_dir / filename

    # Avoid collisions
    counter = 1
    while filepath.exists():
        filepath = domain_dir / f"{slug}-{date_str}-{counter}.md"
        counter += 1

    now_str = datetime.now().strftime("%Y-%m-%d")
    meta = {
        "title": title,
        "author": agent,
        "created": now_str,
        "updated": now_str,
        "confidence": round(confidence, 2),
        "verification_status": "auto_distilled",
        "data_freshness": now_str,
        "decay_class": "normal",
        "sources_count": len(source_episode_ids),
        "domain": domain,
        "tags": tags,
        "distilled_from": source_episode_ids[:10],
        "distillation_date": now_str,
    }

    try:
        import knowledge_utils
        knowledge_utils.write_frontmatter(filepath, meta, body)
    except ImportError:
        # Fallback: manual frontmatter writing
        lines = ["---"]
        for key, value in meta.items():
            if isinstance(value, (list, dict)):
                lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
            else:
                lines.append(f"{key}: {value}")
        lines.extend(["---", "", body.rstrip(), ""])
        filepath.write_text("\n".join(lines), encoding="utf-8")

    return filepath


# ── Main distillation orchestrator ───────────────────────────────────────────

class EpisodicDistiller:
    """Orchestrates the episodic → semantic distillation pipeline."""

    def __init__(
        self,
        db_path: str,
        knowledge_root: Path,
        agent_name: str = "hermes",
        threshold: int = DEFAULT_THRESHOLD,
        cluster_min: int = DEFAULT_CLUSTER_MIN,
        llm_fn: Any | None = None,
    ):
        """
        Args:
            db_path: Path to trustmem-episodes.sqlite
            knowledge_root: Path to trustmem knowledge/ directory
            agent_name: Agent identifier for attribution
            threshold: Min raw episodes before distillation triggers
            cluster_min: Min episodes in a cluster to distill
            llm_fn: Callable(prompt: str) -> str for LLM summarization.
                     If None, uses a simple extractive fallback.
        """
        self.db_path = db_path
        self.knowledge_root = knowledge_root
        self.agent_name = agent_name
        self.threshold = threshold
        self.cluster_min = cluster_min
        self._llm_fn = llm_fn

    def should_distill(self) -> bool:
        """Check if enough raw episodes have accumulated."""
        return count_raw_episodes(self.db_path, self.agent_name) >= self.threshold

    def run(self, force: bool = False) -> dict[str, Any]:
        """
        Execute one distillation cycle.

        Returns:
            {
                "distilled": int,       # number of knowledge entries created
                "episodes_processed": int,
                "clusters_found": int,
                "files_created": list[str],
                "skipped_reason": str | None,
            }
        """
        result: dict[str, Any] = {
            "distilled": 0,
            "episodes_processed": 0,
            "clusters_found": 0,
            "files_created": [],
            "skipped_reason": None,
        }

        raw_count = count_raw_episodes(self.db_path, self.agent_name)
        if not force and raw_count < self.threshold:
            result["skipped_reason"] = (
                f"only {raw_count} raw episodes (threshold: {self.threshold})"
            )
            return result

        # Step 1: Fetch raw episodes
        episodes = fetch_raw_episodes(self.db_path, agent_id=self.agent_name)
        if not episodes:
            result["skipped_reason"] = "no raw episodes found"
            return result

        # Step 2: Cluster
        clusters = cluster_episodes(episodes, min_cluster_size=self.cluster_min)
        result["clusters_found"] = len(clusters)

        if not clusters:
            result["skipped_reason"] = (
                f"{len(episodes)} episodes but no clusters >= {self.cluster_min}"
            )
            return result

        # Step 3: Distill each cluster
        all_consolidated_ids: list[str] = []

        for cluster in clusters:
            try:
                prompt = build_distillation_prompt(cluster)
                llm_response = self._call_llm(prompt)
                parsed = self._parse_llm_response(llm_response)

                if parsed is None:
                    logger.debug("distiller: failed to parse LLM response for cluster")
                    continue

                episode_ids = [ep["id"] for ep in cluster if ep.get("id")]

                # Check for overlap with existing knowledge before writing
                if self._has_similar_knowledge(parsed["title"], parsed.get("tags", [])):
                    logger.debug("distiller: skipping duplicate knowledge: %s", parsed["title"])
                    all_consolidated_ids.extend(episode_ids)
                    continue

                filepath = write_knowledge_file(
                    knowledge_root=self.knowledge_root,
                    title=parsed["title"],
                    body=parsed["body"],
                    tags=parsed.get("tags", []),
                    confidence=parsed.get("confidence", 0.5),
                    source_episode_ids=episode_ids,
                    agent=self.agent_name,
                )

                result["distilled"] += 1
                result["files_created"].append(str(filepath))
                all_consolidated_ids.extend(episode_ids)

                logger.info(
                    "distiller: created %s from %d episodes",
                    filepath.name, len(cluster),
                )

            except Exception as exc:
                logger.warning("distiller: cluster distillation failed: %s", exc)
                continue

        # Step 4: Mark episodes as consolidated
        if all_consolidated_ids:
            mark_episodes_consolidated(self.db_path, all_consolidated_ids)

        result["episodes_processed"] = len(all_consolidated_ids)
        return result

    def _has_similar_knowledge(self, title: str, tags: list[str]) -> bool:
        """Check if similar knowledge already exists using knowledge_consolidate."""
        try:
            import knowledge_consolidate
            groups = knowledge_consolidate.find_overlapping_groups(threshold=0.6)
            # Check if any existing group title overlaps with our new title
            title_lower = title.lower()
            for group in groups:
                for file_info in group.get("files", []):
                    existing_title = str(file_info.get("title", "")).lower()
                    # Simple overlap check: >50% word overlap
                    title_words = set(re.findall(r'[a-z]{3,}|[\u4e00-\u9fff]{2,}', title_lower))
                    existing_words = set(re.findall(r'[a-z]{3,}|[\u4e00-\u9fff]{2,}', existing_title))
                    if title_words and existing_words:
                        overlap = len(title_words & existing_words) / len(title_words | existing_words)
                        if overlap > 0.5:
                            return True
        except (ImportError, Exception) as exc:
            logger.debug("distiller: overlap check unavailable: %s", exc)
        return False

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for distillation. Falls back to extractive summary."""
        if self._llm_fn is not None:
            return self._llm_fn(prompt)
        return self._extractive_fallback(prompt)

    @staticmethod
    def _extractive_fallback(prompt: str) -> str:
        """
        No-LLM fallback: extract title from first episode, combine summaries.
        Produces valid JSON that _parse_llm_response can consume.
        """
        # Extract episodes from prompt text
        summaries = re.findall(r'Summary:\s*(.+)', prompt)
        if not summaries:
            return json.dumps({
                "title": "Consolidated episodes",
                "body": "Multiple related episodes were consolidated.",
                "tags": [],
                "confidence": 0.4,
            })

        # Use first summary as title seed
        title = summaries[0][:80].strip()
        if len(title) > 60:
            title = title[:57] + "..."

        # Combine all summaries as body
        body_parts = [f"- {s.strip()}" for s in summaries]
        body = (
            f"# {title}\n\n"
            "## Consolidated Observations\n\n"
            + "\n".join(body_parts)
            + "\n\n## Pattern\n\n"
            "These episodes share a common theme and were automatically "
            "consolidated from episodic memory."
        )

        return json.dumps({
            "title": title,
            "body": body,
            "tags": [],
            "confidence": 0.4,
        })

    @staticmethod
    def _parse_llm_response(response: str) -> dict[str, Any] | None:
        """Parse LLM JSON response, handling markdown fences."""
        text = response.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        # Validate required fields
        if not isinstance(data, dict) or "title" not in data or "body" not in data:
            return None

        return {
            "title": str(data["title"]),
            "body": str(data["body"]),
            "tags": data.get("tags", []) if isinstance(data.get("tags"), list) else [],
            "confidence": float(data.get("confidence", 0.5)),
        }
