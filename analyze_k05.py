"""k05 (N9) analysis -- implements experiments/PREREGISTER_k05_lowdata.md exactly.

Criteria 1, 2, 4 and 5 were committed before this run existed; criterion 3 runs through
`analyze_n4.py` unchanged. Anything not in that file is exploratory and is labelled so.

  Q1  the failures are dataset size, not algebra   grok >=2/3 at frac 0.80, monotone
  Q2  C6 at 7^2, on a model that actually learned  |Gini_mult - 0.559| < 0.10, >=2/3
  Q4  C8 replicates at 54 and 63                   perm p < 0.01 in >=2/3 seeds, both
  #5  the 0.30 anchor reproduces k04's failure

No timing claim is made here, and none may be added: phi(n) is 18-42 at these moduli and
steps/phi(n) is strongly predicted by phi(n) itself (rho = +0.875), which is what retracted
C19. Grok rate is the outcome, not grok time.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.tasks.algebra import describe
from src.analysis.sparsity import gini
from src.analysis.transforms import unit_index
from src import provenance
import test_crt_law as T

D = sys.argv[1] if len(sys.argv) > 1 else "results/k05_lowdata"
MOD, FRACS, SEEDS = [49, 54, 63], [0.30, 0.50, 0.80], [0, 1, 2]
CRT_TESTABLE = [54, 63]                 # 49 = 7^2 is VACUOUS for C8 by construction
BASELINE = 0.559                        # k02 cyclic mean, the same number P4 used
K04_ANCHOR = {49: 0, 54: 1, 63: 1}      # grokked seeds out of 3 at frac 0.30


def tag(n, f, s):
    return f"B_thesis_n{n}_f{int(f * 100):02d}_s{s}"


def emb(n, f, s):
    p = f"{D}/WE_{tag(n, f, s)}.npz"
    if not os.path.exists(p):
        return None
    W = np.load(p, allow_pickle=False)["W_E"][:n].astype(float)
    return W - W.mean(0, keepdims=True)


def gini_mult(n, f, s):
    """Discrete-log reordering, then the additive DFT over phi(n) points. Cyclic only."""
    idx, orders, _ = unit_index(n)
    if len(orders) != 1:
        return None
    p = f"{D}/WE_{tag(n, f, s)}.npz"
    if not os.path.exists(p):
        return None
    Ud = np.array(sorted(idx, key=lambda x: idx[x]))
    W = np.load(p, allow_pickle=False)["W_E"][:n].astype(float)
    Wu = W[Ud] - W[Ud].mean(0, keepdims=True)
    return gini(T.freq_energy(Wu, len(Ud)))


def load():
    p = f"{D}/lowdata_summary.json"
    if not os.path.exists(p):
        sys.exit(f"NOT LANDED: {p}")
    return json.load(open(p))


def grok_table(rows):
    """{(n, frac): [grok_step or None per seed]} -- the only outcome this run reports."""
    g = {}
    for r in rows:
        g.setdefault((r["n"], round(r["train_frac"], 2)), []).append(r["grok_step"])
    return g


def main():
    rows = load()
    g = grok_table(rows)
    print("=" * 78)
    print("k05 -- critical dataset size at 49, 54 and 63")
    print("=" * 78)
    print(f"\n{len(rows)} runs present of the 27 pre-registered")

    print("\n" + "-" * 78)
    print("Q1 / criterion 1 -- grok RATE by train_frac (not grok time; see the header)")
    print("-" * 78)
    print(f"  {'n':>5} {'phi':>5} " + "".join(f"{'frac ' + str(f):>14}" for f in FRACS))
    rate = {}
    for n in MOD:
        cells = []
        for f in FRACS:
            gs = g.get((n, f), [])
            ok = sum(x is not None for x in gs)
            rate[(n, f)] = (ok, len(gs))
            cells.append(f"{ok}/{len(gs)}" if gs else "--")
        print(f"  {n:>5} {describe(n)['phi']:>5} " + "".join(f"{c:>14}" for c in cells))
    at80 = all(rate.get((n, 0.80), (0, 0))[0] >= 2 for n in MOD)
    mono = all(rate.get((n, FRACS[i]), (0, 0))[0] <= rate.get((n, FRACS[i + 1]), (0, 0))[0]
               for n in MOD for i in range(len(FRACS) - 1))
    print(f"  >=2/3 at frac 0.80 for every modulus : {'YES' if at80 else 'NO'}")
    print(f"  grok rate monotone in train_frac     : {'YES' if mono else 'NO'}")
    print(f"  -> Q1 {'HELD' if at80 and mono else 'NOT HELD'}"
          + ("" if at80 and mono else "   (the failure is NOT purely dataset size)"))

    print("\n" + "-" * 78)
    print("criterion 5 -- does the frac 0.30 anchor reproduce k04's failure?")
    print("-" * 78)
    ok_anchor = True
    for n in MOD:
        got, tot = rate.get((n, 0.30), (None, 0))
        exp = K04_ANCHOR[n]
        same = got == exp
        ok_anchor &= same
        print(f"  n={n:<4} frac 0.30: {got}/{tot} grokked, k04 saw {exp}/3   "
              f"{'MATCHES' if same else 'DIFFERS -- the comparison is not clean'}")
    print(f"  -> anchor {'HELD' if ok_anchor else 'NOT HELD'}")

    print("\n" + "-" * 78)
    print("Q2 / criterion 2 -- C6 at 49 = 7^2, on seeds that actually grokked")
    print("-" * 78)
    frac49 = next((f for f in FRACS if rate.get((49, f), (0, 0))[0] >= 2), None)
    if frac49 is None:
        print("  49 does not grok in >=2/3 seeds at ANY train_frac -- Q2 CANNOT BE ASKED.")
        print("  That is a dataset-size null, NOT evidence against C6. Report it as such.")
    else:
        print(f"  smallest train_frac where 49 groks in >=2/3 seeds: {frac49}")
        grokked = [s for s, x in zip(SEEDS, g[(49, frac49)]) if x is not None]
        vals, dl = [], []
        for s in grokked:
            v = gini_mult(49, frac49, s)
            if v is None:
                continue
            vals.append(v); dl.append(abs(v - BASELINE))
        ok = sum(d < 0.10 for d in dl)
        print(f"  Gini_mult per grokked seed : {[f'{v:.3f}' for v in vals]}")
        print(f"  |delta| vs {BASELINE}        : {[f'{d:.3f}' for d in dl]}")
        print(f"  within 0.10 in {ok}/{len(dl)} grokked seeds (predicted >=2)")
        print(f"  -> Q2 {'HELD -- the clock survives at 7^2' if ok >= 2 else 'NOT HELD'}")
        if ok < 2:
            print("     A REAL NEGATIVE FOR C6: a grokked prime power that is not sparse")
            print("     in the multiplicative basis. C6 needs a restriction, and 7^2")
            print("     differs from 11^2, 5^3, 3^4 and 13^2 in no way we have identified.")

    print("\n" + "-" * 78)
    print("Q4 / criterion 4 -- C8 at the two CRT-testable moduli, frac 0.80")
    print("-" * 78)
    hold = 0
    for n in CRT_TESTABLE:
        es, ps = [], []
        for s in SEEDS:
            W = emb(n, 0.80, s)
            if W is None:
                continue
            r = T.permutation_test(T.freq_energy(W, n), [k - 1 for k in T.predicted(n)])
            if r is None:
                continue
            es.append(r[0]); ps.append(r[1])
        ok = sum(p < 0.01 for p in ps)
        hold += ok >= 2
        print(f"  n={n:<4} enrichment {[f'{x:.1f}x' for x in es]}   p<0.01 in {ok}/{len(ps)}")
    print(f"  holds at {hold} of {len(CRT_TESTABLE)}; predicted >=2 of 2")
    print(f"  -> Q4 {'HELD' if hold >= 2 else 'NOT HELD'}")
    print("  (49 = 7^2 is a prime power: VACUOUS for C8 by construction, not tested)")

    print("\n" + "-" * 78)
    print("criterion 6 -- provenance: real git SHA AND the account that ran it")
    print("-" * 78)
    bad = []
    for r in rows:
        p = f"{D}/WE_{tag(r['n'], r['train_frac'], r['seed'])}.npz"
        try:
            pr = provenance.read(p) or {}
        except Exception:
            pr = {}
        sha, acct = pr.get("git_sha", "unknown"), pr.get("kaggle_account", "unknown")
        if sha in ("unknown", "", None) or sha.startswith("__") or acct.startswith("__"):
            bad.append((p, sha, acct))
    accts = {(provenance.read(f"{D}/WE_{tag(r['n'], r['train_frac'], r['seed'])}.npz")
              or {}).get("kaggle_account") for r in rows[:1]}
    print(f"  {len(bad)} of {len(rows)} artifacts lack a clean SHA or account")
    print(f"  account that ran this: {accts}")


if __name__ == "__main__":
    main()
