---
name: research-line-l3-synthesis
description: "Bi-weekly synthesis of the NOUS OS Research Line evidence base. Reads the last 14 days of L2-promoted notes, session reviews, and prior synthesis; produces docs/research-line/synthesis/YYYY-MM-DD.md as a PR. Never auto-merges."
version: 1.0.0
author: NOUS OS
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [nous-os, research-line, synthesis, bi-weekly, github, pr]
    related_skills: [research-line-l2-triage]
---

# Research Line · L3 Bi-weekly Synthesis

You are running the **L3 bi-weekly synthesis** tier of the NOUS OS Research Line. The L2 weekly triage has been promoting inbound notes for the past 2 weeks. Your job is to read everything that period produced — inbound notes, session reviews, the prior synthesis — and write a synthesis document. You open a PR; the operator gates the merge.

**Authority:** the canonical contract lives at `docs/research-line/hermes-integration.md` § L3 in jupiturliu/nous-os. The synthesis template lives at `docs/research-line/synthesis/_template.md`. **If anything in this skill conflicts with the spec, the spec wins.** Re-read both before each run.

## When you are invoked

Hermes' cron fires you **every Sunday at 14:00 UTC**. The skill itself self-gates to bi-weekly cadence: on each fire, it checks whether a synthesis file dated within the last 13 days exists, and exits cleanly if so. The effective cadence is **one synthesis every two weeks**.

The job's `workdir` is set to a local checkout of `jupiturliu/nous-os` at `master`.

First scheduled run: the first Sunday at 14:00 UTC after the operator runs `scripts/register-cron.sh`. The operator may invoke manually at any time; the self-gate prevents accidental duplicates within a 13-day window.

## What you read

In order:

1. `docs/research-line/hermes-integration.md` § L3 — re-read the spec section.
2. `docs/research-line/synthesis/_template.md` — the format you target.
3. `docs/research-line/research-line.md` § 2 — the two near-term instruments (a) and (b).
4. All `docs/research-line/inbound/YYYY-MM-DD-*.md` files dated **within the last 14 days** (excluding `_inbox/`).
5. All session review packets within the last 14 days:
   - Sandbox: `04 Reviews/Student Sandbox v1 Trial Review *.md` in the Obsidian mirror; or `docs/student-sandbox-v1-review-template.md`-shaped files committed to the repo if any.
   - Trading-agent: outcome review packets (location TBD per Codex; if absent, note that in the synthesis).
6. The previous synthesis, if it exists: `docs/research-line/synthesis/<prev>.md`. (For the first synthesis, this is absent — note that explicitly.)
7. `docs/research-line/anchor-atlas.md` — to count atlas additions and read the bucket distribution.

You may **not** read raw `_inbox/` files; those are not part of the public corpus. Only L2-promoted notes count as inputs to the synthesis.

## What you produce

Exactly one new file:

```
docs/research-line/synthesis/YYYY-MM-DD.md
```

The date is the Sunday on which the synthesis is produced (e.g. `2026-06-07.md`). No other files are modified by this skill.

The file follows `synthesis/_template.md`. Required sections, **in this exact order**:

1. `## 1 · Period at a glance` — numeric snapshot of the last 14 days, no editorial.
2. `## 2 · Most influential inbound notes` — 1–3 notes from the period, each with "what we learned" + "what we changed because of it".
3. `## 3 · Coverage observations` — source/keyword/bucket health for the period.
4. `## 4 · Instrument signals` — direction signals for (a) and (b); honest paragraph on (c).
5. `## 5 · What we were wrong about` — first-class negative results from pre-registrations completed in the period (if any).
6. `## 6 · Next period's planned shifts` — feed for the next round of pre-registrations.

## Hard boundaries (L3-specific; in addition to L2's)

1. **Never auto-merge** the PR.
2. **Never edit existing synthesis files.** New file per period.
3. **Never make causal claims that exceed the evidence base.** At N ≤ 5 sessions for a sub-line, the strongest allowable claim is `"direction signal, not validation"`. Use that phrase verbatim where appropriate. At bi-weekly cadence, N will almost always be small — be especially disciplined.
4. **Never use** `validated`, `confirmed`, `proven` about (a)/(b)/(c) instruments in any synthesis section.
5. **Never extrapolate Sandbox findings to trading-agent or vice versa.** Different sub-lines have different unit definitions. Cross-pollination requires an explicit "and this transfers because…" paragraph naming the assumption.
6. **Never recommend dropping a method commitment** (the 6 listed in `research-line.md` § 4). They are durable by design.
7. **You may propose** changes to the source list, keyword list, or atlas bucket structure in § 3 / § 6. You may **not implement** those changes in the same PR — structural changes ship as separate operator-reviewed PRs.
8. **Never modify** `research-line.md`, `anchor-atlas.md`, `research-line-atlas.html`, `cron-design.md`, or any inbound note. Synthesis is a read-only-of-other-docs activity.

## Step-by-step workflow

### 1. Bi-weekly self-gate

```bash
TODAY=$(date -u +%Y-%m-%d)
# Find the most recent synthesis file (excluding the template).
LATEST=$(ls -1 docs/research-line/synthesis/*.md 2>/dev/null \
  | grep -v _template | sort | tail -1)

if [ -n "${LATEST}" ]; then
  LATEST_DATE=$(basename "${LATEST}" .md)
  # macOS / BSD date: -j -f for parsing; Linux: -d. Use python for portability.
  DAYS_SINCE=$(python3 -c "
from datetime import datetime
a = datetime.strptime('${TODAY}', '%Y-%m-%d')
b = datetime.strptime('${LATEST_DATE}', '%Y-%m-%d')
print((a - b).days)
")
  if [ "${DAYS_SINCE}" -lt 13 ]; then
    echo "Skipping: most recent synthesis ${LATEST_DATE} is only ${DAYS_SINCE} days old (< 13)."
    exit 0
  fi
fi

SYNTH_DATE="${TODAY}"
echo "Synthesizing: period ending ${SYNTH_DATE}"
```

The 13-day threshold gives ~bi-weekly cadence while tolerating cron jitter (a Sunday that fires 12 hours late is still ≥ 13 days from the prior Sunday's synthesis).

### 2. Bootstrap

```bash
git status
git pull --ff-only origin master

cat docs/research-line/hermes-integration.md
cat docs/research-line/synthesis/_template.md
cat docs/research-line/research-line.md
```

### 3. Collect the period's evidence

```bash
# Last 14 days (use the synthesis date as the right edge).
CUTOFF=$(python3 -c "
from datetime import datetime, timedelta
d = datetime.strptime('${SYNTH_DATE}', '%Y-%m-%d') - timedelta(days=14)
print(d.strftime('%Y-%m-%d'))
")

# Find inbound notes promoted in the last 14 days.
for f in $(find docs/research-line/inbound -maxdepth 1 -name "*.md" -not -name "_*" | sort); do
  basename_date=$(basename "$f" .md | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2}' || echo "")
  if [ -n "${basename_date}" ] && [ "${basename_date}" \> "${CUTOFF}" ]; then
    echo "=== $f ==="
    cat "$f"
  fi
done

# Session review packets — Sandbox.
find . -name "Student Sandbox v1 Trial Review*.md" 2>/dev/null
# trading-agent outcome reviews — location TBD, search broadly.
find . -name "*outcome*review*.md" -o -name "*reviewed*experiment*.md" 2>/dev/null \
  | head -20

# Prior synthesis (for continuity).
[ -n "${LATEST}" ] && cat "${LATEST}"
```

### 4. Draft the synthesis

Open a new file `docs/research-line/synthesis/${SYNTH_DATE}.md` and write the six sections.

Key authoring notes:

- **§ 1 (Period at a glance):** Count, don't editorialize. "2 L2 promotions" not "a strong fortnight for promotions."
- **§ 2 (Most influential):** Pick 1–3 from the period's promoted notes. For each, name the specific doc/test/sub-line decision it changed. If nothing concretely changed, say so and explain why we kept the note.
- **§ 3 (Coverage observations):** Bucket distribution; sources with zero promotions in the period; new candidate sources; keyword tweaks. At bi-weekly cadence, do not over-react to one quiet period.
- **§ 4 (Instrument signals):** Be honest about N. At bi-weekly cadence, N is almost always tiny. Default phrasing: "no session evidence in this period" or "1 session contributed data; direction signal, not validation."
- **§ 5 (What we were wrong about):** Pull from pre-registration files in `docs/research-line/preregistration/` whose `captured` date is within the period. If no completed pre-registrations in the period, say so explicitly.
- **§ 6 (Next period's planned shifts):** Concrete proposals. These should feed the next 2 weeks' pre-registrations and L2 triage focus.

### 5. Self-check

```bash
# Linter — required section headers present and in order?
python3 -c "
import re, sys
content = open('docs/research-line/synthesis/${SYNTH_DATE}.md').read()
required = [
  '## 1 · Period at a glance',
  '## 2 · Most influential inbound notes',
  '## 3 · Coverage observations',
  '## 4 · Instrument signals',
  '## 5 · What we were wrong about',
  '## 6 · Next period\\'s planned shifts',
]
positions = [content.find(h) for h in required]
if any(p < 0 for p in positions):
  print('FAIL: missing section'); sys.exit(1)
if positions != sorted(positions):
  print('FAIL: sections out of order'); sys.exit(1)
print('OK')
"

# Forbidden-word scan.
if grep -E "\bvalidated\b|\bconfirmed\b|\bproven\b" "docs/research-line/synthesis/${SYNTH_DATE}.md"; then
  echo "FAIL: synthesis must not claim validation/confirmation/proof at this N"
  exit 1
fi

# Existing tests.
python3 -m unittest discover -s tests 2>&1 | tail -5
```

If any check fails, **fix it before opening the PR**.

### 6. Open the PR

```bash
BRANCH="research-line/l3-synthesis-${SYNTH_DATE}"

git config user.name "nous-os-hermes"
git config user.email "hermes@nousos.ai"

git checkout -b "$BRANCH"
git add "docs/research-line/synthesis/${SYNTH_DATE}.md"
git commit -m "L3 synthesis · ${SYNTH_DATE}" \
  -m "Bi-weekly synthesis of the NOUS OS Research Line evidence base for the 14-day period ending ${SYNTH_DATE}. See file body for sections and PR body for review notes."

git push --force-with-lease origin "$BRANCH"

gh label create "research-line:l3-synthesis" --color "00d4aa" --description "L3 bi-weekly synthesis PR opened by Hermes" 2>/dev/null || true

gh pr create \
  --base master \
  --head "$BRANCH" \
  --title "L3 synthesis · ${SYNTH_DATE}" \
  --label "research-line:l3-synthesis" \
  --body "$(cat <<EOF
Bi-weekly Research Line L3 synthesis produced by Hermes skill \`research-line-l3-synthesis\`.

## Period
14 days ending ${SYNTH_DATE}

## What this PR contains
- \`docs/research-line/synthesis/${SYNTH_DATE}.md\` — the synthesis document

## Operator actions
- **Read** all six sections. § 4 (Instrument signals) and § 5 (What we were wrong about) are load-bearing.
- **Edit** prose to your voice; the operator owns the final framing.
- **Verify** Hermes did not claim validation/confirmation/proof of any instrument (text was scanned but human read is the gate).
- **Merge** to publish — synthesis appears at \`/docs/research-line/synthesis/${SYNTH_DATE}.md\` on nousos.ai.

This synthesis does NOT modify any existing doc. Source-list / keyword / atlas changes proposed in § 3 or § 6 must ship as **separate** operator-reviewed PRs.
EOF
)"
```

### 7. Output summary

```
L3 synthesis complete:
  Period: 14 days ending ${SYNTH_DATE}
  PR: https://github.com/jupiturliu/nous-os/pull/<N>
  Inbound notes in scope: <count>
  Session reviews in scope: <count>
  Most influential picks: <count>
```

## Acceptable outcomes

- **Synthesis with 1–3 influential notes, modest instrument signals, honest negative results.** Common case after a productive fortnight.
- **Synthesis with 0–1 influential notes and "no instrument signal yet" sections.** Common for early bi-weeks with low N — bi-weekly cadence means most periods will have small numbers; that is expected.
- **Synthesis flagging "no session reviews yet, no instrument data".** Honest for periods where Phase B (real Sandbox trials) has not yet started. Synthesize what we read; do not pretend evidence exists.

## Unacceptable outcomes

- A synthesis that overclaims at low N. If you find yourself wanting to write "validated", "confirmed", "proven", or "we have shown" — stop, rewrite to "direction signal, not validation" or "one cycle suggests, not enough to conclude".
- A synthesis that proposes structural changes to research-line.md or anchor-atlas.md within this same PR. Propose in § 6, do not implement.
- A synthesis that crosses sub-lines without an explicit transfer-assumption paragraph.

## Failure modes

- **No inbound notes in the period.** Synthesize what little there is. Flag in § 3 that L2 promotion volume was zero and propose adjustments in § 6 — but do not over-react to a single quiet bi-week.
- **No session reviews in the period.** Synthesize without that input; explicitly say so in § 4. Do **not** fabricate.
- **Forbidden-word scan triggered.** Rewrite the offending sentence; do not commit until the scan passes.
- **`gh pr create` rate-limited:** retry after 60s; if still failing, save the draft and exit with an error.
- **Self-gate hit (< 13 days since last synthesis):** clean exit. This is correct behavior, not failure.
