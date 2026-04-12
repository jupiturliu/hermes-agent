# TrustMem — Hermes Memory Provider Plugin

Bridges [TrustMem](https://github.com/jupiturliu/trustmem)'s trust-weighted episodic + knowledge memory into the Hermes agent lifecycle.

## Features

- **Knowledge search** — semantic search over verified, trust-scored AI research (Python, no API required)
- **Episodic memory** — logs every conversation turn and session summary as a retrievable episode
- **Four-tier visibility** — `private | team | domain | global` episode access control
- **Two agent tools** — `trustmem_search` and `trustmem_reason` exposed to the LLM
- **No MCP server required** — Python tools imported directly; `trustmem reason` invoked via Node subprocess

## Setup

```bash
# 1. Point to the trustmem repo
export TRUSTMEM_ROOT=~/ClaudeCowork/trustmem

# 2. (Optional) custom episode DB path
export TRUSTMEM_EPISODE_DB=~/.hermes/trustmem-episodes.sqlite

# 3. (Optional) agent name for episode attribution
export TRUSTMEM_AGENT=hermes
```

Or write config via `save_config()`:

```python
provider.save_config(
    {"trustmem_root": "/path/to/trustmem"},
    hermes_home=os.path.expanduser("~/.hermes"),
)
```

## Configuration

| Key | Env var | Default | Required |
|-----|---------|---------|----------|
| `trustmem_root` | `TRUSTMEM_ROOT` | auto-detect | Yes |
| `trustmem_episode_db` | `TRUSTMEM_EPISODE_DB` | `~/.hermes/trustmem-episodes.sqlite` | No |
| `trustmem_agent` | `TRUSTMEM_AGENT` | `hermes` | No |
| `trustmem_model_mode` | `TRUSTMEM_MODEL_MODE` | `auto` | No |

## Hooks

| Hook | Purpose |
|------|---------|
| `prefetch` | Synchronous recall on first turn; background on subsequent turns |
| `sync_turn` | Log each turn as an episode (non-blocking daemon thread) |
| `on_session_end` | Write session summary episode after conversation ends |
| `on_pre_compress` | Snapshot last 3 user turns before context compression |
| `on_delegation` | Log subagent task completions in parent's episode store |
| `on_memory_write` | Mirror MEMORY.md writes into episodic store |

## Tools

### `trustmem_search`
```json
{ "query": "CXL memory pooling", "top": 5, "layer": "all" }
```
Returns ranked results from knowledge base + episodic memory with confidence scores.

### `trustmem_reason`
```json
{ "question": "When did we decide to use SGLang?", "top_k": 5 }
```
Answers temporal/factual questions using the episodic reasoning engine (requires built CLI).
