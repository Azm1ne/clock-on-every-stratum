"""Which Gate 2 criteria actually SEPARATE a grokked model from a failed one?

  PYTHONPATH=. .venv/bin/python scripts/gate2_discriminate.py

The G2 pre-registration named **G1 (permutation p) as the PRIMARY criterion**, on the
strength of it being the project's established statistic (C22/C23/C32). Section 4 then
named its own falsifier:

    "The negative-control arm passing G1. A failed run must not show a local clock. If
     n125_s1 (acc 0.6082) reads the same as a grokked run, the statistic is measuring the
     grid, not the model."

IT FIRED. This script is what turns that from a sentence into a number: it scores every
criterion on the grokked arms and on the FAILED arm and prints both rates side by side.

Why G1 is confounded, stated plainly: **30 % of every block is training data**, and a model
that has merely MEMORISED its training cells still has logits that are a function of
u*v there -- so the diagonal still carries the energy and removing it still hurts more than
removing a random bin set. G1 tests "the logits are a function of the product", which
memorisation satisfies. G0 (held-out accuracy), G2 (excluded/baseline) and G5 (neuron
tuning) are what separate the two.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_gate2 import G0_ACC, G1_P, G2_EXCL, G3_RATIO, G4_R, G5_MARGIN  # noqa: E402

TESTS = {
    "G0 held-out acc >= 0.95":   lambda s: s["acc_heldout"] >= G0_ACC,
    "G1 perm p < 0.01":          lambda s: s["p_perm"] < G1_P,
    "G2 excluded/base >= 100":   lambda s: s["excluded"] / max(s["baseline"], 1e-30) >= G2_EXCL,
    "G3 resid <= 0.5 x ctrl":    lambda s: (None if s.get("resid") is None
                                            else s["resid"] <= G3_RATIO * s["resid_ctrl"]),
    "G4 R >= 5":                 lambda s: s["R"] >= G4_R,
    "G5 tuned - shuffled >= .2": lambda s: (None if s.get("tuned") is None
                                            else s["tuned"] - s["tuned_sh"] >= G5_MARGIN),
}


def load(pat):
    out = []
    for f in glob.glob(pat):
        d = json.load(open(f))
        for run in d["runs"]:
            for s in run["strata"]:
                if s["measurable"]:
                    out.append(s)
    return out


def main():
    grokked = load("results/gate2/k03_grid_acts.json") + load("results/gate2/n7_engine.json")
    failed = load("results/gate2/*_controls.json")
    if not failed:
        print("no FAILED-arm artifact yet -- run analyze_gate2.py <dir> --controls")
        return
    print(f"GROKKED stratum-measurements: {len(grokked)}   "
          f"FAILED stratum-measurements: {len(failed)}\n")
    print(f"  {'criterion':<28} {'grokked pass':>14} {'FAILED pass':>13}   separates?")
    print("  " + "-" * 72)
    for name, fn in TESTS.items():
        g = [fn(s) for s in grokked]; g = [x for x in g if x is not None]
        b = [fn(s) for s in failed];  b = [x for x in b if x is not None]
        if not g or not b:
            print(f"  {name:<28} {'n/a':>14} {'n/a':>13}")
            continue
        gr, br = sum(g) / len(g), sum(b) / len(b)
        verdict = ("YES" if gr > 0.9 and br < 0.25 else
                   "partly" if gr > 0.9 and br < 0.75 else "**NO**")
        print(f"  {name:<28} {sum(g):>5}/{len(g):<4} {gr:>6.0%} "
              f"{sum(b):>4}/{len(b):<4} {br:>5.0%}   {verdict}")
    print("\n  A criterion that passes on FAILED runs is a NECESSARY condition, not evidence.")
    print("  G1 was pre-registered as PRIMARY; if it reads **NO** above, the verdict's weight")
    print("  rests on the criteria that do separate, and the write-up must say so.")


if __name__ == "__main__":
    main()
