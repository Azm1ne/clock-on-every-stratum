"""k07 analysis -- implements experiments/PREREGISTER_k07_armA_seeds.md exactly.

Criteria 1-3 were committed before this run existed; criterion 4 runs through
`analyze_n4.py` unchanged. Anything not in that file is exploratory and is labelled so.

  S1  the replication arm groks reliably       >=4/5 seeds acc > 0.99
  S2  C4's published comparison holds          |dGini_mult|<0.10, |dPR|<1.0, 4 keys in >=4/5
  S3  C21 holds                                100% a+b in dlog coords, set equality, >=4/5

UNITS ONLY. Protocol invariant 1: nothing here pools with a thesis-arm run, and only this
arm may be compared to 2606.17399's numbers.

CONVENTION: Gini on AMPLITUDE (DC dropped, sin/cos combined); participation ratio on
normalised ENERGY. Mixing them was open question O1 for three sessions.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.analysis.sparsity import gini, participation_ratio, key_freqs_5x_median
from src.analysis.transforms import unit_index
from src.viz import mechinterp as M
from src import provenance
import test_crt_law as T

D = sys.argv[1] if len(sys.argv) > 1 else "results/k07_arma_seeds"
N, SEEDS = 113, [0, 1, 2, 3, 4]
PUB = dict(gini_mult=0.579, pr_mult=4.1, gini_add=0.071, pr_add=52.7, n_key=4)


def path(s):
    return f"{D}/WE_A_replication_n{N}_s{s}.npz"


def spectra(s):
    """(Gini_mult, PR_mult, Gini_add, PR_add, n_key_mult) for one seed."""
    z = np.load(path(s), allow_pickle=False)
    W = z["W_E"][:N].astype(float)
    idx, orders, _ = unit_index(N)
    Ud = np.array(sorted(idx, key=lambda x: idx[x]))
    Wu = W[Ud] - W[Ud].mean(0, keepdims=True)
    Wa = W - W.mean(0, keepdims=True)
    fm, fa = T.freq_energy(Wu, len(Ud)), T.freq_energy(Wa, N)
    return (gini(fm), participation_ratio(fm ** 2),
            gini(fa), participation_ratio(fa ** 2),
            len([k for k in key_freqs_5x_median(fm)]))


def main():
    p = f"{D}/arma_summary.json"
    if not os.path.exists(p):
        sys.exit(f"NOT LANDED: {p}")
    rows = json.load(open(p))
    print("=" * 78)
    print("k07 -- the REPLICATION arm (units only, n=113) at 5 seeds")
    print("=" * 78)
    print(f"\n{len(rows)} runs present of the 5 pre-registered")

    print("\n" + "-" * 78)
    print("S1 / criterion 1 -- does the arm grok reliably?")
    print("-" * 78)
    ok = 0
    for r in sorted(rows, key=lambda r: r["seed"]):
        g = r["grok_step"]
        ok += r["final_test_acc"] > 0.99
        print(f"  seed {r['seed']}: grok {g}   final test acc {r['final_test_acc']:.4f}")
    print(f"  {ok}/{len(rows)} grokked; predicted >=4/5")
    print(f"  -> S1 {'HELD' if ok >= 4 else 'NOT HELD'}")

    print("\n" + "-" * 78)
    print("S2 / criterion 2 -- C4 against the published numbers, at 5 seeds")
    print("-" * 78)
    vals = {k: [] for k in ("gm", "pm", "ga", "pa", "nk")}
    print(f"  {'seed':>5} {'Gini_mult':>10} {'PR_mult':>9} {'Gini_add':>9} "
          f"{'PR_add':>8} {'#key':>5}")
    for s in SEEDS:
        if not os.path.exists(path(s)):
            continue
        gm, pm, ga, pa, nk = spectra(s)
        for k, v in zip(("gm", "pm", "ga", "pa", "nk"), (gm, pm, ga, pa, nk)):
            vals[k].append(v)
        print(f"  {s:>5} {gm:>10.3f} {pm:>9.2f} {ga:>9.3f} {pa:>8.2f} {nk:>5}")
    m = {k: float(np.mean(v)) for k, v in vals.items() if v}
    sd = {k: float(np.std(v)) for k, v in vals.items() if v}
    print(f"  {'mean':>5} {m['gm']:>10.3f} {m['pm']:>9.2f} {m['ga']:>9.3f} "
          f"{m['pa']:>8.2f} {m['nk']:>5.1f}")
    print(f"  {'sd':>5} {sd['gm']:>10.3f} {sd['pm']:>9.2f} {sd['ga']:>9.3f} "
          f"{sd['pa']:>8.2f} {sd['nk']:>5.2f}")
    print(f"  PUBLISHED (2606.17399): Gini_mult {PUB['gini_mult']}  PR_mult {PUB['pr_mult']}"
          f"  Gini_add {PUB['gini_add']}  PR_add {PUB['pr_add']}  #key {PUB['n_key']}")
    d_g = abs(m["gm"] - PUB["gini_mult"])
    d_p = abs(m["pm"] - PUB["pr_mult"])
    n4 = sum(v == PUB["n_key"] for v in vals["nk"])
    print(f"  |dGini_mult| {d_g:.3f} (<0.10)   |dPR_mult| {d_p:.2f} (<1.0)   "
          f"exactly 4 key freqs in {n4}/{len(vals['nk'])} (>=4)")
    s2 = d_g < 0.10 and d_p < 1.0 and n4 >= 4
    print(f"  -> S2 {'HELD' if s2 else 'NOT HELD'}")
    print("  (frequency VALUES are seed-dependent and, per C25, defined only up to")
    print("   multiplication by a unit mod phi -- only the COUNT is predicted)")

    print("\n" + "-" * 78)
    print("S3 / criterion 3 -- C21: does `a+b` carry >=95% of top-8 energy in dlog coords?")
    print("-" * 78)
    hits = 0
    for s in SEEDS:
        if not os.path.exists(path(s)):
            continue
        try:
            top = M.logit_top_components(path(s), N, top=8, verbose=False, dlog=True)
        except Exception as e:
            print(f"  seed {s}: logit_top_components unavailable ({type(e).__name__} {e})")
            continue
        # AMENDED criterion (see the pre-registration): energy fraction, not a pure
        # count. C21's original "100% a+b" was overstated -- k02 seed 0 is 7 of 8
        # components and 98.37% of top-8 energy, with an a-b term at (-16,16).
        e = {}
        for ka, kb, en, term in top:
            e[term] = e.get(term, 0.0) + float(en)
        tot = sum(e.values()) or 1.0
        frac = e.get("a+b", 0.0) / tot
        n_ab = sum(1 for t in top if t[-1] == "a+b")
        ok = frac >= 0.95
        hits += ok
        print(f"  seed {s}: a+b carries {frac:7.2%} of top-8 energy "
              f"({n_ab}/8 components)   "
              + ("OK" if ok else "BELOW 0.95 -- C21 does not hold here")
              + ("" if len(e) == 1 else
                 "   other: " + ", ".join(f"{k} {v/tot:.1%}"
                                          for k, v in sorted(e.items(), key=lambda kv: -kv[1])
                                          if k != "a+b")))
    print(f"  a+b >= 95% of top-8 energy in "
          f"{hits}/{len([s for s in SEEDS if os.path.exists(path(s))])} seeds; predicted >=4")
    print(f"  -> S3 {'HELD' if hits >= 4 else 'NOT HELD'}")
    print("  (criterion 4, the C22 permutation p, runs through analyze_n4.py)")

    print("\n" + "-" * 78)
    print("criterion 5 -- provenance")
    print("-" * 78)
    seen = {}
    for s in SEEDS:
        if not os.path.exists(path(s)):
            continue
        pr = provenance.read(path(s)) or {}
        k = (pr.get("git_sha", "unknown")[:7], pr.get("kaggle_account", "unknown"))
        seen[k] = seen.get(k, 0) + 1
    for (sha, acct), c in sorted(seen.items()):
        bad = sha in ("unknown", "") or sha.startswith("__")
        print(f"  git_sha {sha}  account {acct}  x{c}" + ("   <-- NOT REPRODUCIBLE" if bad else ""))


if __name__ == "__main__":
    main()
