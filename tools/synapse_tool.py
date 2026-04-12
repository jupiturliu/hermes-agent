"""Synapse Agent Fabric tool — dispatch tasks to AI Factory agents via Event Bus.

Registers synapse_dispatch as a Hermes tool that routes tasks through
Synapse's pub/sub Event Bus instead of P2P delegate_task.

3-file pattern:
  1. This file: tool implementation + registry.register()
  2. model_tools.py: add "tools.synapse_tool" to _modules
  3. toolsets.py: add "synapse_dispatch" to a toolset
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Synapse imports (lazy) ───────────────────────────────────────────────

_bridge = None
_workers_started = False

SYNAPSE_DIR = Path.home() / "ClaudeCowork" / "synapse"


def _get_bridge():
    global _bridge, _workers_started

    if _bridge is not None:
        return _bridge

    # Add Synapse to path
    synapse_path = str(SYNAPSE_DIR)
    if synapse_path not in sys.path:
        sys.path.insert(0, synapse_path)

    try:
        # Suppress Redis warnings
        import builtins
        _orig = builtins.print
        builtins.print = lambda *a, **k: _orig(*a, **{**k, 'file': sys.stderr})

        from orchestration.hermes_synapse_bridge import HermesSynapseBridge
        from hermes_workers import HERMES_AGENTS, start_workers

        builtins.print = _orig

        _bridge = HermesSynapseBridge(backend="memory", agent_id="hermes_tool")

        if not _workers_started:
            start_workers(list(HERMES_AGENTS.keys()), backend="memory")
            _workers_started = True
            time.sleep(0.5)
            logger.info("Synapse: started %d workers", len(HERMES_AGENTS))

        return _bridge
    except Exception as exc:
        logger.warning("Synapse not available: %s", exc)
        return None


# ── Tool handler ─────────────────────────────────────────────────────────

SYNAPSE_DISPATCH_SCHEMA = {
    "name": "synapse_dispatch",
    "description": (
        "Dispatch a task to another AI Factory agent via the Synapse Event Bus. "
        "This routes through pub/sub (O(N)) instead of P2P delegate_task (O(N²)). "
        "Available agents: architect, briefing, research, simulation, hardware, "
        "inference, cost, fault, dsx, fleet, power, cooling, thermal, idc, "
        "network, storage, dt_engineer, execution."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "Agent short name: research, simulation, hardware, inference, "
                    "cost, fault, dsx, fleet, briefing, power, cooling, thermal, "
                    "idc, network, storage, dt_engineer, execution"
                ),
            },
            "task": {
                "type": "string",
                "description": "Natural language task for the target agent",
            },
            "context": {
                "type": "string",
                "description": "Optional JSON context (source, URLs, data)",
            },
            "wait": {
                "type": "boolean",
                "description": "Wait for result (true) or return job_id immediately (false). Default true.",
            },
        },
        "required": ["target", "task"],
    },
}


def synapse_dispatch(target: str, task: str, context: str = "", wait: bool = True, **kwargs) -> str:
    """Dispatch a task to an AI Factory agent via Synapse."""
    bridge = _get_bridge()
    if bridge is None:
        return json.dumps({"error": "Synapse not available. Check ~/ClaudeCowork/synapse/ exists."})

    # Parse context
    ctx = {}
    if context:
        try:
            ctx = json.loads(context)
        except json.JSONDecodeError:
            ctx = {"raw": context}

    job_id = bridge.dispatch(target, task, ctx)

    if not wait:
        return json.dumps({"job_id": job_id, "status": "dispatched", "target": target})

    result = bridge.wait(job_id, timeout_s=120)
    return json.dumps(result, ensure_ascii=False, default=str)


# ── Blackboard tools ─────────────────────────────────────────────────────

SYNAPSE_BLACKBOARD_SCHEMA = {
    "name": "synapse_blackboard",
    "description": (
        "Read or write shared state on the Synapse Blackboard. "
        "Agents use this to share real-time data (simulation results, fault status, POD config) "
        "without P2P calls. Action: 'read' or 'write'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write"],
                "description": "read or write",
            },
            "agent_id": {
                "type": "string",
                "description": "Agent who owns the data",
            },
            "scope": {
                "type": "string",
                "description": "Data scope (e.g., 'result', 'context', 'config')",
            },
            "data": {
                "type": "string",
                "description": "JSON data to write (only for write action)",
            },
        },
        "required": ["action", "agent_id", "scope"],
    },
}


def synapse_blackboard(action: str, agent_id: str, scope: str, data: str = "", **kwargs) -> str:
    bridge = _get_bridge()
    if bridge is None:
        return json.dumps({"error": "Synapse not available"})

    if action == "read":
        result = bridge.blackboard.read(agent_id, scope)
        return json.dumps({"agent_id": agent_id, "scope": scope, "data": result}, ensure_ascii=False, default=str)
    elif action == "write":
        try:
            parsed = json.loads(data) if data else {}
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON data"})
        bridge.blackboard.write_diff(agent_id, scope, parsed)
        return json.dumps({"status": "written", "agent_id": agent_id, "scope": scope})
    else:
        return json.dumps({"error": f"Unknown action: {action}"})


# ── Registration ─────────────────────────────────────────────────────────

from tools.registry import registry

registry.register(
    name="synapse_dispatch",
    toolset="synapse",
    schema=SYNAPSE_DISPATCH_SCHEMA,
    handler=lambda args, **kw: synapse_dispatch(
        target=args.get("target", ""),
        task=args.get("task", ""),
        context=args.get("context", ""),
        wait=args.get("wait", True),
    ),
    emoji="🔀",
)

registry.register(
    name="synapse_blackboard",
    toolset="synapse",
    schema=SYNAPSE_BLACKBOARD_SCHEMA,
    handler=lambda args, **kw: synapse_blackboard(
        action=args.get("action", "read"),
        agent_id=args.get("agent_id", ""),
        scope=args.get("scope", ""),
        data=args.get("data", ""),
    ),
    emoji="📋",
)
