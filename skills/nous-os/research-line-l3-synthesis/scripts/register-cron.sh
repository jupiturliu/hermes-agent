#!/usr/bin/env bash
# Register the L3 bi-weekly synthesis cron job in Hermes.
#
# Run once on the Hermes operator machine. Idempotent — re-running this
# script refreshes the schedule but does not duplicate the job (Hermes
# keys jobs by name in ~/.hermes/cron/jobs.json).
#
# The skill self-gates to bi-weekly cadence: it checks whether a synthesis
# file dated within the last 13 days exists, and exits cleanly if so.
# So the cron can fire every Sunday safely; the skill produces output
# once every two weeks.
#
# Prerequisites: same as research-line-l2-triage/scripts/register-cron.sh

set -euo pipefail

NOUS_OS_CHECKOUT="${NOUS_OS_CHECKOUT:-/Users/liyao/nousos/nous-os}"

if [ ! -d "${NOUS_OS_CHECKOUT}" ]; then
  echo "Error: NOUS_OS_CHECKOUT=${NOUS_OS_CHECKOUT} does not exist."
  exit 1
fi

# Cron: every Sunday at 14:00 UTC (after the 12:00 UTC L2 triage).
# Skill itself enforces bi-weekly cadence via the 13-day self-gate.
hermes cron add \
  --name "nous-os-research-line-l3-synthesis" \
  --schedule "0 14 * * 0" \
  --skill "nous-os:research-line-l3-synthesis" \
  --workdir "${NOUS_OS_CHECKOUT}" \
  --prompt "Run the bi-weekly L3 synthesis of the NOUS OS Research Line per the skill instructions. The skill self-gates to one synthesis per ~14 days; if a synthesis for the current bi-week already exists, exit cleanly. Otherwise open a PR titled 'L3 synthesis · YYYY-MM-DD' in jupiturliu/nous-os." \
  --deliver "local"

echo ""
echo "Registered L3 synthesis cron (every Sunday 14:00 UTC; skill self-gates to bi-weekly)."
echo "Verify with:"
echo "  hermes cron list"
