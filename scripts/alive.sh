#!/usr/bin/env bash
# Is a python script running? Prints its PID, or nothing.
#
#   scripts/alive.sh analyze_gate2.py
#   until [ -z "$(scripts/alive.sh foo.py)" ]; do sleep 20; done
#
# WHY THIS EXISTS. `pgrep -f foo.py` and `awk '$0 ~ /foo/'` match THE MATCHING COMMAND'S OWN
# ARGV. In this project that has killed the agent's own shell three times, made two waiters
# spin on a condition that could never become false, made a watchdog miscount, and on
# 2026-09-14 stalled a fifth waiter -- written while the gotcha was on screen in LAB_PROTOCOL.md.
#
# Two rules, and both are needed:
#   1. match the script name as a WHOLE FIELD, never a substring. `awk -v s=foo.py` puts
#      `s=foo.py` in this process's own argv, which is not equal to `foo.py`, so the scan
#      cannot see itself.
#   2. require argv[0] to be a python interpreter, so the `bash -c "... foo.py ..."` wrapper
#      that launched it does not match either.
# And never index to a fixed field: `$4=="foo.py"` is right only until someone adds an
# interpreter flag, which is how session_brief.py and STATE.md both reported the box idle
# while a 20-hour sweep was 10.5 hours in.
[ $# -eq 1 ] || { echo "usage: $0 <script.py>" >&2; exit 2; }
# `ps -eo pid=,args=` puts the PID in $1 and argv[0] in $2, so the interpreter guard is $2
# and the scan starts at $3. Getting that off by one made the first version of this script
# report a live 40k-step run as idle -- a FALSE IDLE, the direction that makes the next
# session relaunch on top of a running job. Verified against a live job, not by reading it.
ps -eo pid=,args= | awk -v s="$1" '
  $2 ~ /python[0-9.]*$/ { for (i = 3; i <= NF; i++) if ($i == s) { print $1; break } }'
