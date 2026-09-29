#!/usr/bin/env bash
# Every Gate 2 arm, in order, each to its own log. One process, so the arms cannot race.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
P=.venv/bin/python
$P -u analyze_gate2.py                              > logs/gate2_k03_final.log  2>&1
$P -u analyze_gate2.py results/n7_engine            > logs/gate2_engine.log      2>&1
$P -u analyze_gate2.py results/n7_engine --controls > logs/gate2_ctrl_engine.log 2>&1
$P -u analyze_gate2.py results/k04_extended --controls > logs/gate2_ctrl_k04.log 2>&1
echo FINISHED > logs/gate2_arms.status
