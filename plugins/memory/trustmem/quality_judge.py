"""
TrustMem Episode Quality Judge
===============================
Scores the quality of agent interactions before they are persisted as
episodic memories. Inspired by Databricks MemAlign's LLM judge approach:
only high-quality episodes are worth memorizing.

Two-tier scoring:
  1. **Heuristic scorer** (fast, zero-cost) — rule-based signals
  2. **LLM judge** (optional) — deeper semantic evaluation for borderline cases

Configuration:
  TRUSTMEM_QUALITY_THRESHOLD — min score to persist (default: 0.3)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_QUALITY_THRESHOLD = 0.3
LLM_BORDERLINE_LOW = 0.25
LLM_BORDERLINE_HIGH = 0.45


@dataclass
class QualityVerdict:
    """Result of quality evaluation."""
    score: float
    outcome: str
    signals: dict[str, float] = field(default_factory=dict)
    should_persist: bool = True


# ── Heuristic signal functions ───────────────────────────────────────────────

_ERROR_PATTERNS = re.compile(
    r'(?i)(?:error|exception|traceback|failed|failure|cannot|unable to|'
    r'not found|timed?\s*out|permission denied|401|403|404|500|'
    r'ImportError|KeyError|ValueError|TypeError|RuntimeError|'
    r'ENOENT|EACCES|EPERM)',
)

_TOOL_PATTERNS = re.compile(
    r'(?i)(?:tool_call|function_call|```(?:python|bash|json|sql)|'
    r'created file|modified file|wrote to|executed|running test|'
    r'PASSED|FAILED|assert)',
)

_TRIVIAL_PATTERNS = re.compile(
    r'(?i)^(?:ok|okay|sure|got it|understood|yes|no|thanks|thank you|'
    r'sounds good|will do|let me know|hello|hi|hey)[\s.!?]*$',
)

_INSIGHT_PATTERNS = re.compile(
    r'(?i)(?:because|therefore|the reason|key insight|important|'
    r'recommend|suggest|should|must|critical|decision|trade-?off|'
    r'pattern|learned|discovered|root cause|conclusion|takeaway)',
)


def _signal_content_length(user: str, assistant: str) -> float:
    total = len(user) + len(assistant)
    if total < 20:
        return 0.0
    if total < 100:
        return 0.2
    if total < 500:
        return 0.5
    if total < 2000:
        return 0.7
    return 0.85


def _signal_specificity(user: str, assistant: str) -> float:
    combined = user + " " + assistant
    specificity = 0.0
    if '```' in combined or 'def ' in combined or 'class ' in combined:
        specificity += 0.3
    if re.search(r'[/\\][\w.-]+\.\w{1,5}', combined):
        specificity += 0.15
    if re.search(r'\d+\.\d+|\d{3,}|0x[0-9a-f]+', combined):
        specificity += 0.1
    if re.search(r'https?://', combined):
        specificity += 0.1
    tech_terms = len(re.findall(
        r'(?i)(?:api|sql|http|json|yaml|docker|git|cpu|gpu|'
        r'memory|cache|thread|async|deploy|config|schema|'
        r'function|module|class|method|variable|parameter)',
        combined,
    ))
    specificity += min(0.35, tech_terms * 0.05)
    return min(1.0, specificity)


def _signal_error_presence(assistant: str) -> float:
    matches = len(_ERROR_PATTERNS.findall(assistant))
    if matches == 0:
        return 0.0
    if matches <= 2:
        return 0.3
    return min(1.0, matches * 0.15)


def _signal_tool_usage(assistant: str) -> float:
    matches = len(_TOOL_PATTERNS.findall(assistant))
    if matches == 0:
        return 0.0
    return min(1.0, matches * 0.2)


def _signal_trivial(user: str, assistant: str) -> float:
    if _TRIVIAL_PATTERNS.match(user.strip()):
        return 0.8
    if _TRIVIAL_PATTERNS.match(assistant.strip()):
        return 0.5
    if len(user.strip()) < 10 and len(assistant.strip()) < 50:
        return 0.6
    return 0.0


def _signal_insight(assistant: str) -> float:
    matches = len(_INSIGHT_PATTERNS.findall(assistant))
    if matches == 0:
        return 0.0
    return min(1.0, matches * 0.15)


def _signal_cjk_content(user: str, assistant: str) -> float:
    combined = user + assistant
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff]', combined))
    if cjk_chars == 0:
        return 0.0
    ratio = cjk_chars / max(len(combined), 1)
    return 0.15 if ratio > 0.3 else 0.05


# ── Heuristic scorer ─────────────────────────────────────────────────────────

_SIGNAL_WEIGHTS = {
    "content_length": 0.20,
    "specificity":    0.25,
    "tool_usage":     0.20,
    "insight":        0.20,
    "cjk_content":    0.05,
    "trivial":       -0.30,
    "error_presence": -0.10,
}


def score_heuristic(user_content: str, assistant_content: str) -> QualityVerdict:
    """Fast rule-based quality scoring. Returns score in [0.0, 1.0]."""
    user = user_content or ""
    assistant = assistant_content or ""

    signals = {
        "content_length": _signal_content_length(user, assistant),
        "specificity":    _signal_specificity(user, assistant),
        "tool_usage":     _signal_tool_usage(assistant),
        "insight":        _signal_insight(assistant),
        "cjk_content":    _signal_cjk_content(user, assistant),
        "trivial":        _signal_trivial(user, assistant),
        "error_presence": _signal_error_presence(assistant),
    }

    raw_score = sum(signals[name] * weight for name, weight in _SIGNAL_WEIGHTS.items())
    score = max(0.0, min(1.0, raw_score))

    error_level = signals["error_presence"]
    if error_level >= 0.6:
        outcome = "failure"
    elif error_level >= 0.3:
        outcome = "partial"
    elif signals["trivial"] >= 0.6:
        outcome = "trivial"
    else:
        outcome = "success"

    return QualityVerdict(
        score=round(score, 3),
        outcome=outcome,
        signals={k: round(v, 3) for k, v in signals.items()},
    )


# ── LLM judge ───────────────────────────────────────────────────────────────

LLM_JUDGE_PROMPT = (
    "Rate this agent interaction for long-term memorization value.\n\n"
    "## Interaction\n"
    "User: {user}\n\n"
    "Assistant: {assistant}\n\n"
    "## Criteria\n"
    "- Does this contain a reusable insight, decision, or discovery?\n"
    "- Would recalling this help the agent in future similar tasks?\n"
    "- Is the content specific enough to be actionable?\n"
    "- Is this more than trivial chat or error noise?\n\n"
    "## Output (JSON only)\n"
    '{{"score": 0.0-1.0, "outcome": "success|partial|failure|trivial", '
    '"reason": "one sentence"}}'
)


def score_llm(
    user_content: str,
    assistant_content: str,
    llm_fn: Callable[[str], str],
) -> QualityVerdict | None:
    """
    LLM-based quality scoring for borderline episodes.
    Returns None if LLM call fails (caller should fall back to heuristic).
    """
    prompt = LLM_JUDGE_PROMPT.format(
        user=user_content[:500],
        assistant=assistant_content[:500],
    )
    try:
        response = llm_fn(prompt)
        return _parse_llm_verdict(response)
    except Exception as exc:
        logger.debug("quality_judge llm error: %s", exc)
        return None


def _parse_llm_verdict(response: str) -> QualityVerdict | None:
    """Parse LLM judge response into QualityVerdict."""
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(data, dict) or "score" not in data:
        return None

    score = float(data["score"])
    score = max(0.0, min(1.0, score))
    outcome = data.get("outcome", "success")
    if outcome not in ("success", "partial", "failure", "trivial"):
        outcome = "success"

    return QualityVerdict(score=round(score, 3), outcome=outcome)


# ── Combined judge ───────────────────────────────────────────────────────────

def judge_episode(
    user_content: str,
    assistant_content: str,
    threshold: float = DEFAULT_QUALITY_THRESHOLD,
    llm_fn: Callable[[str], str] | None = None,
) -> QualityVerdict:
    """
    Two-tier quality gate:
      1. Always run heuristic scorer (free)
      2. If score is borderline AND llm_fn is provided, run LLM judge
      3. Apply threshold to decide should_persist

    Returns QualityVerdict with should_persist set.
    """
    verdict = score_heuristic(user_content, assistant_content)

    # If borderline and LLM judge available, get second opinion
    if (llm_fn is not None
            and LLM_BORDERLINE_LOW <= verdict.score <= LLM_BORDERLINE_HIGH):
        llm_verdict = score_llm(user_content, assistant_content, llm_fn)
        if llm_verdict is not None:
            # Blend: 40% heuristic + 60% LLM (LLM is more reliable for borderline)
            blended = verdict.score * 0.4 + llm_verdict.score * 0.6
            verdict = QualityVerdict(
                score=round(blended, 3),
                outcome=llm_verdict.outcome,
                signals=verdict.signals,
            )

    verdict.should_persist = verdict.score >= threshold
    return verdict
