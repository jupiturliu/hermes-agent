---
name: research-line-l2-triage
description: "Weekly triage of the NOUS OS Research Line L1 inbox. Promotes 0–3 raw entries to full 1-page inbound notes. Opens a PR; never auto-merges."
version: 1.0.0
author: NOUS OS
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [nous-os, research-line, triage, weekly, github, pr]
    related_skills: [github, arxiv]
---

# Research Line · L2 Weekly Triage

You are running the **L2 weekly triage** tier of the NOUS OS Research Line. The L1 capture cron has been producing daily raw `_inbox/YYYY-MM-DD.md` files for the past week. Your job is to read them, judge which 1–3 entries deserve full 1-page inbound notes, draft those notes, update the anchor atlas, and open a PR for the human operator to review and merge.

**Authority:** the canonical contract for this work lives at `docs/research-line/hermes-integration.md` § L2 in the jupiturliu/nous-os repo. **The boundaries below are reproduced from that spec; if they conflict with anything you remember, the spec wins.** Re-read it before each run if you have any doubt.

## When you are invoked

Hermes' cron fires you Sunday 12:00 UTC. The job's `workdir` is set to a local checkout of `jupiturliu/nous-os`. The current branch is `master`, up to date with origin.

You may be invoked manually for ad-hoc triage; the workflow is identical.

## What you read

In order:

1. `docs/research-line/hermes-integration.md` — re-read the L2 spec section.
2. `docs/research-line/research-line.md` — the line's positioning.
3. `docs/research-line/anchor-atlas.md` — the curated map. Note which anchors are currently `queued`, `scanned`, `note-written`.
4. `docs/research-line/inbound/_inbox/*.md` — last 7 days of capture (excluding today, since today's capture may not yet be operator-merged).
5. `docs/research-line/inbound/_template.md` — the format every new note must follow.
6. The three seed inbound notes at `docs/research-line/inbound/2026-05-17-*.md` — exemplars of the target voice and depth.

Read with `Read` / `cat` only. **Never** make HTTP requests to "verify" or "expand" entries. L1 already captured the snapshot; that is the canonical input.

## Selection rubric (apply to every candidate)

| Criterion | Pass | Demote / fail |
|---|---|---|
| Distinctness | Adds a new anchor, angle, or evidence | Substantially overlaps with an existing `note-written` anchor |
| Anchor specificity | Maps cleanly to one of the 6 atlas buckets | Fits 3+ buckets equally well (too vague) |
| Defensibility | You can write a credible "where we differ / what we add" line | Differentiation would be hand-wavy → leave as `scanned`, do not promote |
| Freshness | Published in the past 14 days | Older → only promote with explicit reasoning in PR body |
| Source diversity | Different source than last week's promotions | Same source repeating → prefer another bucket |

**Cap: 3 per week.** If 5 score equally, pick 3 and explain in the PR body. **If 0 pass, that is the correct answer.** Open the PR with no inbound changes and a body explaining what was reviewed and why nothing rose to the bar.

## Hard boundaries (do not violate)

1. **Never auto-merge** the PR. The operator gates every promotion.
2. **Never edit existing inbound notes.** New notes only.
3. **Never modify structural sections** of `research-line.md`, `anchor-atlas.md`, or `research-line-atlas.html` (north star, sub-lines, method commitments, bucket headings). Only append to bucket bodies and flip anchor status pills.
4. **Never refer to the operator** by name, family, or identifying detail in any draft note. If a candidate entry contains identifying detail, surface it in the PR body as a flag and exclude the entry.
5. **Never make HTTP requests** to verify or expand a candidate. Work from L1 capture only.
6. **Never claim an anchor is "definitively positioned"** — `note-written` means *we have written down our positioning*, not that the positioning is final.
7. **Never use** the words `validated`, `confirmed`, or `proven` about NOUS OS's own (a)/(b)/(c) instruments in any draft note. Those words are reserved for outputs of session evidence, not your prose.
8. **Cap at 3 promotions per week.** No exceptions; if you find 5 worth promoting, the discipline is to pick 3.

## Step-by-step workflow

### 1. Bootstrap

```bash
# Confirm we are in a clean nous-os checkout on master.
git status
git log --oneline -1
git pull --ff-only origin master

# Enumerate the last 7 days of inbox files. We exclude today because the
# operator may not have merged today's L1 PR yet.
ls -1 docs/research-line/inbound/_inbox/*.md 2>/dev/null \
  | tail -8 | head -7

# Read the spec.
cat docs/research-line/hermes-integration.md
```

If `inbound/_inbox/` is empty or has fewer than 3 files, the cron probably missed days. Still proceed — write the 0-promotions PR with a coverage flag.

### 2. Read and score

For each inbox file, read it fully (`cat docs/research-line/inbound/_inbox/YYYY-MM-DD.md`). Build an internal list of all unique entries (deduplicate by URL across days). For each entry, apply the rubric. Keep a short written rationale per scored entry — it informs the PR body.

### 3. Pick 0–3 promotions

Order remaining candidates by your overall judgment. Take the top 0–3. If you can't honestly write a "where we differ" line for an entry, do not promote it — that is a `scanned` candidate, not a `note-written` candidate.

### 4. Draft new inbound notes

For each promotion, create a file at:

```
docs/research-line/inbound/YYYY-MM-DD-<short-slug>.md
```

where `<short-slug>` is hyphen-separated lowercase (e.g., `mollick-2026-04-15`, `dwarkesh-ep-217-sutskever`).

Follow `docs/research-line/inbound/_template.md` exactly. Required sections, in order:

1. YAML frontmatter (title, authors, year, venue, kind, status: `note-written`, captured, anchor_bucket)
2. `## What it is`
3. `## Why it matters for our line`
4. `## Where we share`
5. `## Where we differ / what we add` — this is the **load-bearing** section
6. `## What this changes in our practice`
7. `## Limitations of this work (from our perspective)`
8. `## Open questions for follow-up`
9. `## Citation`

If you cannot fill section 5 in a defensible way, do not promote.

### 5. Update the atlas (both md and html)

For each promoted anchor:

**In `docs/research-line/anchor-atlas.md`:**
- Find the anchor's existing entry. If it exists, flip `**Status:** scanned` (or `queued`) to `**Status:** note-written` and add the link.
- If the anchor is new, append it to the appropriate bucket section with full 4-line structure (What / Claim / Share / Differ) + `**Status:** note-written` + link.

**In `research-line-atlas.html`:**
- Find the matching `<article class="anchor">` block. Change its class to `anchor note`, change `<span class="status …">…</span>` to `<span class="status note">note-written</span>`, and add a new `<dt>Note</dt><dd><a href="/docs/research-line/inbound/...">…</a></dd>` row.
- If the anchor is new, add a new `<article>` block in the matching `<section>`'s `.anchor-grid`.

Use `sed`, `Edit`, or careful manual edits. Do **not** touch other entries.

### 6. Verify locally before opening the PR

```bash
# Run the existing contract tests to catch obvious breakage.
python3 -m unittest discover -s tests -v 2>&1 | tail -5

# Re-verify atlas spec/html alignment.
python3 -m unittest tests.test_nous_os.BenchmarkTests.test_research_line_atlas_spec_and_web_are_aligned -v
```

If anything fails, fix or back out the relevant change. **Do not commit broken state.**

### 7. Branch, commit, open the PR

```bash
DATE_TAG=$(date -u +%Y-%m-%d)
BRANCH="research-line/l2-triage-${DATE_TAG}"

git config user.name "nous-os-hermes"
git config user.email "hermes@nousos.ai"

git checkout -b "$BRANCH"
git add docs/research-line/inbound/ docs/research-line/anchor-atlas.md research-line-atlas.html
git commit -m "L2 triage · week of ${DATE_TAG}" \
  -m "Promotes <N> inbound candidates to full notes. See PR body for selection rationale."

git push --force-with-lease origin "$BRANCH"
```

Then open the PR:

```bash
gh pr create \
  --base master \
  --head "$BRANCH" \
  --title "L2 triage · week of ${DATE_TAG}" \
  --label "research-line:l2-triage" \
  --body "$(cat <<'EOF'
Weekly Research Line L2 triage produced by Hermes skill `research-line-l2-triage`.

## What was reviewed
- N inbox files spanning YYYY-MM-DD..YYYY-MM-DD
- M unique candidate entries (after de-duplication)

## Selection rationale (top of mind, not in committed files)
1. [Why entry 1 was promoted]
2. [Why entry 2 was promoted]
3. [Why entry 3 was promoted, or why fewer / none were]

## What this PR contains
- `docs/research-line/inbound/YYYY-MM-DD-<slug>.md` × N — new full 1-page notes
- `docs/research-line/anchor-atlas.md` — anchor status pills flipped to `note-written`
- `research-line-atlas.html` — visual mirror updated

## Operator actions
- **Read** the new notes. Each "Where we differ" section is load-bearing.
- **Edit** prose if needed.
- **Reject** by closing this PR if Hermes mis-judged. Add a one-line reason in the close comment.
- **Merge** to publish — CI green → Cloudflare deploys → notes live on nousos.ai.

This PR does NOT auto-merge. Hermes is operator-gated per `docs/research-line/hermes-integration.md` § L2 boundaries.
EOF
)"
```

If the `research-line:l2-triage` label does not exist, create it first:

```bash
gh label create "research-line:l2-triage" --color "7b61ff" --description "L2 weekly triage PR opened by Hermes" || true
```

### 8. Output summary

After the PR is opened, print a short summary for the cron output:

```
L2 triage complete:
  PR: https://github.com/jupiturliu/nous-os/pull/<N>
  Promotions: <count>
  Inbox files reviewed: <count>
  Unique candidates: <count>
```

## Acceptable outcomes

- **3 promotions in the PR.** Common case in a high-signal week.
- **1–2 promotions.** Common case in a normal week.
- **0 promotions, PR still opened with rationale.** Acceptable; the week was lean.
- **0 promotions, no PR.** Acceptable only when there are *zero* operator-merged inbox files this week.

## Failure modes

- **Branch already exists:** the previous week's PR was not merged. Use a date-suffixed branch (e.g., `research-line/l2-triage-2026-05-25-v2`) and explain in the PR body.
- **`gh pr create` rate-limited:** retry once after 60s; if still failing, exit with an error and let the next cron run pick up.
- **Contract test fails after edits:** back out the relevant edit, log it in the PR body's "Issues encountered" section, and proceed with the remaining valid promotions.
- **All inbox files are empty/error-only:** open the PR with 0 promotions and a "coverage" flag in the body; this is a signal for the next L3 synthesis to investigate sources.

## What you don't do (re-emphasized)

- You do **not** edit existing inbound notes or the research-line spec.
- You do **not** decide what counts as a "valid" source — that is `cron-design.md`'s job.
- You do **not** make the merge decision — that is the operator's job.
- You do **not** make claims about (a)/(b)/(c) instruments being validated.
- You do **not** run more than once per week without manual invocation.
