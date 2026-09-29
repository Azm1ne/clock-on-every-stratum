#!/usr/bin/env bash
# Wait for the N7 sweep to finish, then run the whole analysis chain unattended.
#
# Exists so N7 *finishes* rather than merely stopping training while someone sleeps.
# Runs as its own systemd --user unit, so it outlives the terminal and the agent session
# that started it.
#
#   systemd-run --user --unit=n7-finish --working-directory=$PWD \
#     --property=StandardOutput=append:$PWD/logs/n7_finish.log \
#     --property=StandardError=append:$PWD/logs/n7_finish.log bash scripts/n7_finish.sh
#
# Every stage runs even if an earlier one fails -- a crash in one criterion must not cost
# the other three. Each stage's exit code is recorded and the summary lists them, so the
# morning read is "what succeeded", never "it stopped somewhere, find out where".
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${N7_OUT:-results/n7_engine}
POLL=300
say() { echo "[$(date -Is)] $*"; }

say "n7-finish armed; waiting for n7-sweep to finish"
# DEADLOCK FIXED 2026-09-14. `is-active` ALONE IS A CONDITION THAT CAN NEVER BECOME FALSE
# here: the documented launch command sets RemainAfterExit=yes, which keeps the unit
# "active" forever after run_n7_all.sh exits. On 2026-09-12 the sweep completed all 12 runs
# and this loop polled for 21h42m (2.5s CPU over 21h wall) until a logout stopped both
# units -- the analysis never ran and two days were lost with the data sitting on disk.
# Wait on conditions that can actually become true: the unit stops, OR its SubState leaves
# "running" (RemainAfterExit parks a finished unit at active/exited), OR all 12 runs report.
while :; do
  _st=$(systemctl --user is-active n7-sweep 2>/dev/null)
  _sub=$(systemctl --user show n7-sweep -p SubState --value 2>/dev/null)
  _n=$(grep -l "done. grok_step" logs/n7_n*_s*.log 2>/dev/null | wc -l)
  [ "$_st" != "active" ] && { say "sweep unit left active ($_st)"; break; }
  [ "$_sub" = "exited" ] && { say "sweep script exited (SubState=exited, RemainAfterExit)"; break; }
  [ "$_n" -ge 12 ] && { say "all 12 runs reported done"; break; }
  sleep "$POLL"
done

state=$(systemctl --user is-active n7-sweep 2>/dev/null)
result=$(systemctl --user show n7-sweep -p Result --value 2>/dev/null)
done_n=$(grep -lh "done. grok_step" logs/n7_n*_s*.log 2>/dev/null | wc -l)
say "sweep left active: state=$state result=$result  completed runs=${done_n}/12"

# A partial sweep is still worth analysing -- E1/E2/E4/E5 report NOT ASSESSABLE per
# modulus rather than inventing a verdict -- but say so loudly at the top of the log.
if [ "$done_n" -lt 12 ]; then
  say "WARNING: only ${done_n}/12 runs completed. Analysing anyway; verdicts below are"
  say "WARNING: on PARTIAL data and every affected cell should read NOT ASSESSABLE."
  if [ "$result" != "success" ]; then
    say "WARNING: unit result was '$result' -- check logs/n7_sweep.log and, if it was"
    say "WARNING: oom-killed, results/n7_engine/ckpt_*.npz still hold the progress."
  fi
fi

# Every stage is wrapped in `timeout`. An unattended chain whose purpose is to have the
# answer waiting in the morning must not be able to hang silently: a stage that never
# returns would mean waking to no summary at all, which is strictly worse than waking to
# one that says a stage timed out. Limits are generous (these ran in minutes on comparable
# data) -- they are a deadlock guard, not a performance budget. `timeout` reports 124 on
# expiry, which the summary prints as-is.
rc_n7=x; rc_n4=x; rc_rn=x
say "=== analyze_n7.py $OUT   (limit 30m)"
PYTHONPATH=. timeout 1800 .venv/bin/python analyze_n7.py "$OUT"; rc_n7=$?
say "=== analyze_n4.py $OUT   (E3: C22/C23 causality, 200-draw permutations, limit 120m)"
PYTHONPATH=. timeout 7200 .venv/bin/python analyze_n4.py "$OUT"; rc_n4=$?
say "=== scripts/render_all.py   (limit 90m)"
PYTHONPATH=. timeout 5400 .venv/bin/python scripts/render_all.py; rc_rn=$?

say "=============================================================="
say "N7 FINISH SUMMARY   runs ${done_n}/12   sweep result=$result"
say "  analyze_n7  exit $rc_n7"
say "  analyze_n4  exit $rc_n4"
say "  render_all  exit $rc_rn"
say "  groks: $(grep -h 'GROKKED' logs/n7_n*_s*.log 2>/dev/null | sed 's/.*engine_//;s/ \*\*\*//' | tr '\n' ' ')"
say "NEXT (needs judgement, deliberately NOT automated):"
say "  1. fill the Outcome in experiments/PREREGISTER_n7_engine.md from the verdicts above"
say "  2. append a LAB_NOTEBOOK entry"
say "  3. update STATE.md's claims ledger"
say "=============================================================="
