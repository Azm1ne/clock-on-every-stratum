#!/usr/bin/env bash
# The Lean gate. `lake build` exiting 0 is NOT the gate: `sorry` compiles with a warning.
#
#   scripts/lean_gate.sh              build, then check every row of lean/README.md's table
#   scripts/lean_gate.sh --selfcheck  plant a sorry and a native_decide; both must be rejected
#
# A row marked `proved` passes only if `#print axioms` lists nothing beyond propext,
# Classical.choice and Quot.sound. A row marked `stated` must exist (it may use sorry). The
# README table and the theorems in lean/Strata/*.lean must name the same set, and each row's
# file:line must point at its theorem, so the table cannot drift from the source.
set -euo pipefail
cd "$(dirname "$0")/../lean"
export PATH="$HOME/.elan/bin:$PATH"
ALLOWED="propext Classical.choice Quot.sound"
fail=0
trap 'rm -f tmp.*' EXIT

# check_axioms <preamble-file> <name>... : prints one verdict per name, returns 1 on any FAIL
check_axioms() {
  local pre=$1; shift
  local f; f=$(mktemp --suffix=.lean -p .)
  { cat "$pre"; for t in "$@"; do echo "#print axioms $t"; done; } > "$f"
  local out; out=$(lake env lean "$f" 2>&1 | tr '\n' ' ') || true
  rm -f "$f"
  local rc=0
  for t in "$@"; do
    local seg ax a
    seg=$(grep -oP "'\Q$t\E' (does not depend on any axioms|depends on axioms: \[[^]]*\])" <<<"$out" || true)
    if [ -z "$seg" ]; then echo "  [FAIL] $t: no axiom report (missing theorem?)"; rc=1; continue; fi
    ax=$(grep -oP '\[\K[^]]*' <<<"$seg" | tr -d ' ' | tr ',' ' ' || true)
    for a in $ax; do
      if [[ " $ALLOWED " != *" $a "* ]]; then echo "  [FAIL] $t uses $a"; rc=1; continue 2; fi
    done
    echo "  [ok]   $t"
  done
  return $rc
}

if [ "${1:-}" = "--selfcheck" ]; then
  pre=$(mktemp -p .)
  cat > "$pre" <<'EOF'
import Mathlib
theorem plant_ok : (1 : ℕ) + 1 = 2 := rfl
theorem plant_sorry : (1 : ℕ) + 1 = 3 := sorry
theorem plant_native : (2 : ℕ) + 2 = 4 := by native_decide
EOF
  ok=0
  check_axioms "$pre" plant_ok >/dev/null || { echo "SELFCHECK FAILED: the positive control was rejected"; ok=1; }
  if check_axioms "$pre" plant_sorry; then echo "SELFCHECK FAILED: sorry passed"; ok=1; fi
  if check_axioms "$pre" plant_native; then echo "SELFCHECK FAILED: native_decide passed"; ok=1; fi
  if check_axioms "$pre" plant_missing; then echo "SELFCHECK FAILED: a missing theorem passed"; ok=1; fi
  rm -f "$pre"
  [ $ok = 0 ] && echo "SELFCHECK PASSED: control accepted; sorry, native_decide and a missing name rejected"
  exit $ok
fi

lake build 2>&1 | grep -E "^error" && { echo "FAILED: lake build"; exit 1; }
lake build >/dev/null

# Table rows: | ... | `Strata.Name` | `Strata/X.lean:N` | stated|proved |  ->  "Name file line status"
rows=$(sed -nE 's/^\|.*`Strata\.([A-Za-z0-9_]+)` \| `(Strata\/[A-Za-z]+\.lean):([0-9]+)` \| (stated|proved) \|$/\1 \2 \3 \4/p' README.md)
[ -n "$rows" ] || { echo "FAILED: no table rows in lean/README.md"; exit 1; }
if [ "$(cut -d' ' -f1 <<<"$rows" | sort)" != "$(grep -hoP '^theorem \K\w+' Strata/*.lean | sort)" ]; then
  echo "  [FAIL] README table and Strata/*.lean name different theorems:"
  diff <(cut -d' ' -f1 <<<"$rows" | sort) <(grep -hoP '^theorem \K\w+' Strata/*.lean | sort) | sed 's/^/    /' || true
  fail=1
fi
while read -r name file line status; do
  sed -n "${line}p" "$file" | grep -qP "^theorem \Q$name\E\b" || { echo "  [FAIL] $name is not at $file:$line"; fail=1; }
done <<<"$rows"

pre=$(mktemp -p .); echo "import Strata" > "$pre"
proved=$(awk '$4=="proved"{print "Strata."$1}' <<<"$rows")
stated=$(awk '$4=="stated"{print "Strata."$1}' <<<"$rows")
echo "proved (axiom-clean required): $(wc -w <<<"$proved")   stated (sorry allowed): $(wc -w <<<"$stated")"
if [ -n "$proved" ]; then check_axioms "$pre" $proved || fail=1; fi
if [ -n "$stated" ]; then  # must exist; sorry is expected until the row is proved
  out=$(check_axioms "$pre" $stated || true)
  if grep -q "no axiom report" <<<"$out"; then grep "no axiom report" <<<"$out"; fail=1; fi
fi
rm -f "$pre"
[ $fail = 0 ] && echo "PASS: lean gate" || { echo "FAILED: lean gate"; exit 1; }
