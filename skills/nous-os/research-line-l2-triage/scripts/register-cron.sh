#!/usr/bin/env bash
# Register the L2 weekly triage cron job in Hermes.
#
# Run once on the Hermes operator machine. The job is idempotent —
# re-running this script will refresh the schedule but not duplicate
# the job (cron jobs are keyed by name in ~/.hermes/cron/jobs.json).
#
# Prerequisites:
# - hermes CLI installed and on PATH
# - a local checkout of jupiturliu/nous-os at master at the path below
# - gh CLI installed and authenticated as a user with PR-create access to
#   jupiturliu/nous-os (operator's normal gh login is enough)

set -euo pipefail

NOUS_OS_CHECKOUT="${NOUS_OS_CHECKOUT:-/Users/liyao/nousos/nous-os}"

if [ ! -d "${NOUS_OS_CHECKOUT}" ]; then
  echo "Error: NOUS_OS_CHECKOUT=${NOUS_OS_CHECKOUT} does not exist."
  echo "Set NOUS_OS_CHECKOUT to the absolute path of your nous-os checkout."
  exit 1
fi

# Cron: every Sunday at 12:00 UTC.
hermes cron add \
  --name "nous-os-research-line-l2-triage" \
  --schedule "0 12 * * 0" \
  --skill "nous-os:research-line-l2-triage" \
  --workdir "${NOUS_OS_CHECKOUT}" \
  --prompt "Run the weekly L2 triage of the NOUS OS Research Line per the skill instructions. Open a PR titled 'L2 triage · week of YYYY-MM-DD' in jupiturliu/nous-os." \
  --deliver "local"

echo ""
echo "Registered L2 triage cron. Verify with:"
echo "  hermes cron list"
