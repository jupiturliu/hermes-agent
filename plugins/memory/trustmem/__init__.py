"""
TrustMem Hermes MemoryProvider Plugin
======================================
Bridges TrustMem's trust-weighted episodic + knowledge memory into the
Hermes agent lifecycle. Does NOT require the MCP server to be running —
Python tools are imported directly for reads/writes; `trustmem reason`
is invoked via subprocess for LLM-based question answering.

Configuration (env vars or ~/.hermes/trustmem.json):
  TRUSTMEM_ROOT      — path to trustmem repo (required)
  TRUSTMEM_EPISODE_DB — SQLite DB path (optional, default: /tmp/hermes-trustmem.sqlite)
  TRUSTMEM_MODEL_MODE — auto|live|mock (optional)
  TRUSTMEM_AGENT      — agent name used for episode attribution (optional, default: hermes)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── Tool schemas ────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "trustmem_search",
        "description": (
            "Search TrustMem's unified memory: knowledge base (verified, trust-weighted AI research) "
            "and episodic memory (past task outcomes). Returns ranked results with confidence scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — use the core topic or question",
                },
                "top": {
                    "type": "integer",
                    "description": "Max results to return (default 5)",
                    "default": 5,
                },
                "layer": {
                    "type": "string",
                    "enum": ["all", "knowledge", "episodes"],
                    "description": "Which memory layer to search (default: all)",
                    "default": "all",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "trustmem_reason",
        "description": (
            "Answer a specific question using TrustMem's episodic reasoning engine. "
            "Best for temporal questions ('when did I...'), factual recall, or preference queries. "
            "Returns a structured answer with confidence score."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to answer from memory",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Evidence episodes to retrieve (default 5)",
                    "default": 5,
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "trustmem_distill",
        "description": (
            "Distill accumulated episodic memories into semantic knowledge entries. "
            "Clusters related episodes, extracts generalizable patterns/rules, and writes "
            "them to the knowledge base. Run periodically or when you notice the agent repeating "
            "similar discoveries. Returns a summary of what was distilled."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Force distillation even if below threshold (default: false)",
                    "default": False,
                },
            },
            "required": [],
        },
    },
]


# ─── Provider ────────────────────────────────────────────────────────────────

class TrustMemMemoryProvider:
    """Hermes MemoryProvider backed by TrustMem."""

    @property
    def name(self) -> str:
        return "trustmem"

    # ── Availability & init ──────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if TrustMem repo is reachable. No network calls."""
        root = self._resolve_root()
        if root is None:
            return False
        return (root / "tools" / "knowledge_search.py").exists()

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", str(Path.home() / ".hermes"))
        self._agent_context = kwargs.get("agent_context", "primary")
        self._agent_name = os.environ.get("TRUSTMEM_AGENT", "hermes")
        self._root = self._resolve_root()
        self._db_path = os.environ.get(
            "TRUSTMEM_EPISODE_DB",
            str(Path(self._hermes_home) / "trustmem-episodes.sqlite"),
        )

        # Add tools/ to sys.path so we can import directly
        tools_dir = str(self._root / "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)

        # Load Python modules (lazy — only fail loudly on actual call)
        self._ks = None       # knowledge_search module
        self._el = None       # episode_logger module
        self._load_modules()

        # Threading state
        self._prefetch_lock = threading.Lock()
        self._prefetch_cache: str = ""
        self._prefetch_thread: threading.Thread | None = None
        self._sync_thread: threading.Thread | None = None
        self._distill_thread: threading.Thread | None = None
        self._turn_count = 0

        # Skip writes for cron/flush contexts
        self._readonly = self._agent_context in ("cron", "flush")

        # Distiller (lazy-initialized on first use)
        self._distiller = None
        self._distill_threshold = int(os.environ.get("TRUSTMEM_DISTILL_THRESHOLD", "10"))
        self._llm_fn = kwargs.get("llm_fn")  # optional LLM callable from agent

        # Quality judge
        self._quality_threshold = float(os.environ.get("TRUSTMEM_QUALITY_THRESHOLD", "0.3"))
        self._quality_llm = os.environ.get("TRUSTMEM_QUALITY_LLM", "").lower() in ("1", "true", "yes")

    # ── System prompt ────────────────────────────────────────────────────────

    def system_prompt_block(self) -> str:
        """Inject a brief description of TrustMem's scope into the system prompt."""
        if self._readonly:
            return ""
        return (
            "## TrustMem Memory\n"
            "You have access to TrustMem, a trust-weighted memory system with two layers:\n"
            "- **Knowledge base**: verified AI research and team knowledge (with confidence scores)\n"
            "- **Episodic memory**: past task outcomes and observations (your work history)\n"
            "Use `trustmem_search` to recall relevant context, `trustmem_reason` for temporal/factual questions, "
            "or `trustmem_distill` to consolidate accumulated episodes into reusable knowledge."
        )

    # ── Prefetch ─────────────────────────────────────────────────────────────

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return recalled context for this turn (from background thread or sync fallback)."""
        with self._prefetch_lock:
            cached = self._prefetch_cache
            self._prefetch_cache = ""  # consume cache

        if cached:
            return cached

        # First turn: do a synchronous search
        return self._do_search(query, top=5)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire background recall for the next turn."""
        if self._readonly:
            return

        def _work():
            result = self._do_search(query, top=5)
            with self._prefetch_lock:
                self._prefetch_cache = result

        t = threading.Thread(target=_work, daemon=True)
        t.start()
        with self._prefetch_lock:
            self._prefetch_thread = t

    # ── Sync turn ────────────────────────────────────────────────────────────

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Non-blocking: score quality then log the turn if it passes threshold."""
        if self._readonly or not user_content.strip():
            return

        self._turn_count += 1

        def _work():
            try:
                if self._el is None:
                    return
                from plugins.memory.trustmem.quality_judge import judge_episode

                verdict = judge_episode(
                    user_content,
                    assistant_content or "",
                    threshold=self._quality_threshold,
                    llm_fn=self._llm_fn if self._quality_llm else None,
                )

                if not verdict.should_persist:
                    logger.debug(
                        "trustmem: skipping low-quality turn (score=%.2f, outcome=%s)",
                        verdict.score, verdict.outcome,
                    )
                    return

                self._el.log_episode(
                    agent=self._agent_name,
                    task_type="conversation",
                    task=user_content[:500],
                    approach="hermes-session",
                    outcome=verdict.outcome,
                    duration_s=0,
                    quality=verdict.score,
                    notes=assistant_content[:500] if assistant_content else "",
                )
            except Exception as exc:
                logger.debug("trustmem sync_turn error: %s", exc)

        t = threading.Thread(target=_work, daemon=True)
        t.start()
        self._sync_thread = t

    # ── Session end ──────────────────────────────────────────────────────────

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Wait for pending writes, then log a session summary episode and trigger distillation."""
        # Flush pending sync
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=8)

        if self._readonly or not messages or self._el is None:
            return

        # Build a compact session summary
        user_turns = [m["content"] for m in messages if m.get("role") == "user" and m.get("content")]
        if not user_turns:
            return

        summary = f"Session with {len(user_turns)} user turn(s). Topics: {user_turns[0][:200]}"
        try:
            self._el.log_episode(
                agent=self._agent_name,
                task_type="session-summary",
                task=summary,
                approach="hermes-session-end",
                outcome="success",
                duration_s=0,
                quality=0.6,
                notes=f"total turns: {self._turn_count}",
            )
        except Exception as exc:
            logger.debug("trustmem on_session_end error: %s", exc)

        # Trigger background distillation if threshold reached
        self._maybe_distill_async()

    # ── Pre-compress ─────────────────────────────────────────────────────────

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Extract key facts from messages about to be compressed."""
        if self._readonly or not messages or self._el is None:
            return ""

        user_turns = [m["content"] for m in messages if m.get("role") == "user" and m.get("content")]
        if not user_turns:
            return ""

        summary = "; ".join(t[:150] for t in user_turns[-3:])  # last 3 user turns
        try:
            self._el.log_episode(
                agent=self._agent_name,
                task_type="pre-compress",
                task=f"Context about to be compressed: {summary}",
                approach="hermes-compression",
                outcome="success",
                duration_s=0,
                quality=0.5,
            )
        except Exception as exc:
            logger.debug("trustmem on_pre_compress error: %s", exc)

        return ""  # don't inject text into compression summary

    # ── Delegation ───────────────────────────────────────────────────────────

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        """Score and log a subagent completion as an episode."""
        if self._readonly or self._el is None:
            return

        def _work():
            try:
                from plugins.memory.trustmem.quality_judge import judge_episode

                verdict = judge_episode(task, result or "", threshold=self._quality_threshold)
                if not verdict.should_persist:
                    logger.debug("trustmem: skipping low-quality delegation (score=%.2f)", verdict.score)
                    return

                self._el.log_episode(
                    agent=self._agent_name,
                    task_type="delegation",
                    task=task[:500],
                    approach=f"delegated to {child_session_id or 'subagent'}",
                    outcome=verdict.outcome,
                    duration_s=0,
                    quality=verdict.score,
                    notes=result[:500] if result else "",
                )
            except Exception as exc:
                logger.debug("trustmem on_delegation error: %s", exc)

        threading.Thread(target=_work, daemon=True).start()

    # ── Memory write mirror ───────────────────────────────────────────────────

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in MEMORY.md writes into TrustMem episodes with quality scoring."""
        if self._readonly or self._el is None:
            return

        def _work():
            try:
                from plugins.memory.trustmem.quality_judge import score_heuristic

                # Memory writes are intentional — score for metadata but always persist
                verdict = score_heuristic(f"memory {action}: {target}", content)
                self._el.log_episode(
                    agent=self._agent_name,
                    task_type="memory-write",
                    task=f"memory {action}: {content[:300]}",
                    approach=f"builtin-{target}",
                    outcome=verdict.outcome,
                    duration_s=0,
                    quality=max(0.6, verdict.score),  # floor at 0.6 — writes are always valuable
                )
            except Exception as exc:
                logger.debug("trustmem on_memory_write error: %s", exc)

        threading.Thread(target=_work, daemon=True).start()

    # ── Tool schemas & handling ───────────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if tool_name == "trustmem_search":
            return self._tool_search(args)
        if tool_name == "trustmem_reason":
            return self._tool_reason(args)
        if tool_name == "trustmem_distill":
            return self._tool_distill(args)
        return json.dumps({"error": f"unknown tool: {tool_name}"})

    # ── Config schema ─────────────────────────────────────────────────────────

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "trustmem_root",
                "description": "Path to the trustmem repository (e.g. ~/ClaudeCowork/trustmem)",
                "required": True,
                "env_var": "TRUSTMEM_ROOT",
            },
            {
                "key": "trustmem_episode_db",
                "description": "SQLite DB path for episodic memory",
                "required": False,
                "default": "~/.hermes/trustmem-episodes.sqlite",
                "env_var": "TRUSTMEM_EPISODE_DB",
            },
            {
                "key": "trustmem_agent",
                "description": "Agent name for episode attribution",
                "required": False,
                "default": "hermes",
                "env_var": "TRUSTMEM_AGENT",
            },
            {
                "key": "trustmem_model_mode",
                "description": "Embedding/LLM mode: auto|live|mock",
                "required": False,
                "default": "auto",
                "choices": ["auto", "live", "mock"],
                "env_var": "TRUSTMEM_MODEL_MODE",
            },
            {
                "key": "trustmem_distill_threshold",
                "description": "Min raw episodes before auto-distillation triggers (default: 10)",
                "required": False,
                "default": "10",
                "env_var": "TRUSTMEM_DISTILL_THRESHOLD",
            },
            {
                "key": "trustmem_quality_threshold",
                "description": "Min quality score to persist an episode (0.0-1.0, default: 0.3)",
                "required": False,
                "default": "0.3",
                "env_var": "TRUSTMEM_QUALITY_THRESHOLD",
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        """Write non-secret config to $HERMES_HOME/trustmem.json."""
        config_path = Path(hermes_home) / "trustmem.json"
        existing: dict[str, Any] = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def shutdown(self) -> None:
        """Flush pending threads on clean exit."""
        for t in (self._sync_thread, self._prefetch_thread, self._distill_thread):
            if t and t.is_alive():
                t.join(timeout=5)

    # ── Distillation ─────────────────────────────────────────────────────────

    def _get_distiller(self):
        """Lazy-initialize the EpisodicDistiller."""
        if self._distiller is None:
            from plugins.memory.trustmem.distiller import EpisodicDistiller

            knowledge_root = self._root / "knowledge" if self._root else None
            if knowledge_root is None or not knowledge_root.exists():
                return None

            self._distiller = EpisodicDistiller(
                db_path=self.db_path if hasattr(self, 'db_path') else self._db_path,
                knowledge_root=knowledge_root,
                agent_name=self._agent_name,
                threshold=self._distill_threshold,
                llm_fn=self._llm_fn,
            )
        return self._distiller

    def _maybe_distill_async(self) -> None:
        """Trigger background distillation if threshold is reached."""
        def _work():
            try:
                distiller = self._get_distiller()
                if distiller is None:
                    return
                if not distiller.should_distill():
                    return
                result = distiller.run()
                if result["distilled"] > 0:
                    logger.info(
                        "trustmem: auto-distilled %d knowledge entries from %d episodes",
                        result["distilled"], result["episodes_processed"],
                    )
            except Exception as exc:
                logger.debug("trustmem auto-distill error: %s", exc)

        t = threading.Thread(target=_work, daemon=True)
        t.start()
        self._distill_thread = t

    def _tool_distill(self, args: dict[str, Any]) -> str:
        """Handle the trustmem_distill tool call (synchronous)."""
        force = args.get("force", False)
        try:
            distiller = self._get_distiller()
            if distiller is None:
                return json.dumps({"error": "distiller not available (knowledge root not found)"})
            result = distiller.run(force=force)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_root(self) -> Path | None:
        root_env = os.environ.get("TRUSTMEM_ROOT")
        if root_env:
            p = Path(root_env).expanduser()
            return p if p.exists() else None
        # Auto-detect: common locations
        candidates = [
            Path.home() / "ClaudeCowork" / "trustmem",
            Path.home() / "trustmem",
        ]
        for c in candidates:
            if c.exists() and (c / "tools" / "knowledge_search.py").exists():
                return c
        return None

    def _load_modules(self) -> None:
        try:
            import knowledge_search as ks
            import episode_logger as el
            self._ks = ks
            self._el = el
        except ImportError as exc:
            logger.warning("trustmem: could not import Python tools: %s", exc)

    def _do_search(self, query: str, top: int = 5) -> str:
        """Run knowledge search and format results as markdown context."""
        if self._ks is None:
            return ""
        try:
            results = self._ks.search(
                query,
                top_k=top,
                caller=self._agent_name,
                viewer=self._agent_name,
                scope="all",
            )
        except Exception as exc:
            logger.debug("trustmem search error: %s", exc)
            return ""

        if not results:
            return ""

        lines = ["## TrustMem Context"]
        for r in results[:top]:
            title = r.get("title") or r.get("file", "untitled")
            score = r.get("effective_confidence") or r.get("score", 0)
            snippet = (r.get("snippet") or r.get("body", ""))[:300]
            lines.append(f"**{title}** (confidence: {score:.2f})\n{snippet}")
        return "\n\n".join(lines)

    def _tool_search(self, args: dict[str, Any]) -> str:
        query = args.get("query", "")
        top = int(args.get("top", 5))
        layer = args.get("layer", "all")

        if self._ks is None:
            return json.dumps({"error": "knowledge_search not available"})

        try:
            results = self._ks.search(
                query,
                top_k=top,
                caller=self._agent_name,
                viewer=self._agent_name,
                scope="all",
            )
            return json.dumps({"results": results, "query": query, "layer": layer}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def _tool_reason(self, args: dict[str, Any]) -> str:
        question = args.get("question", "")
        top_k = int(args.get("top_k", 5))

        if self._root is None:
            return json.dumps({"error": "TRUSTMEM_ROOT not set"})

        # Strategy 1: Use trustmem search CLI for episode retrieval (reliable)
        trustmem_bin = self._root / "packages" / "episodic-memory" / "dist" / "cli.js"
        if not trustmem_bin.exists():
            return json.dumps({"error": "trustmem CLI not built — run npm run build first"})

        try:
            proc = subprocess.run(
                [
                    "node",
                    str(trustmem_bin),
                    "search",
                    question,
                    "--top", str(top_k),
                    "--db", self._db_path,
                    "--layer", "episodes",
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self._root),
            )
            if proc.returncode != 0:
                return json.dumps({"error": proc.stderr.strip() or "non-zero exit"})

            # Parse search results and format as a reasoning answer
            raw_output = proc.stdout.strip()
            # Remove deprecation warnings from node
            lines = [l for l in raw_output.split("\n") if not l.startswith("(node:")]
            clean_output = "\n".join(lines)

            try:
                episodes = json.loads(clean_output)
            except json.JSONDecodeError:
                return json.dumps({"answer": clean_output, "confidence": 0.5, "source": "raw"})

            if not episodes:
                return json.dumps({"answer": "No relevant episodes found.", "confidence": 0.0, "episodes": []})

            # Build structured answer from top episodes
            evidence = []
            for ep in episodes[:top_k]:
                evidence.append({
                    "id": ep.get("id", ""),
                    "summary": ep.get("title", ""),
                    "details": (ep.get("body") or "")[:500],
                    "score": ep.get("final_score", 0),
                    "date": ep.get("created_at", ""),
                })

            return json.dumps({
                "question": question,
                "evidence": evidence,
                "episode_count": len(evidence),
                "top_score": evidence[0]["score"] if evidence else 0,
            }, ensure_ascii=False)

        except subprocess.TimeoutExpired:
            return json.dumps({"error": "trustmem search timed out"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})


# ─── Registration ─────────────────────────────────────────────────────────────

def register(ctx) -> None:
    """Register TrustMem as a Hermes memory provider plugin."""
    ctx.register_memory_provider(TrustMemMemoryProvider())
