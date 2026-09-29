"""k09 analysis -- implements experiments/PREREGISTER_k09_primes_zdd.md exactly.

  PYTHONPATH=. .venv/bin/python analyze_k09.py [results_dir]
  PYTHONPATH=. .venv/bin/python analyze_k09.py --selfcheck

  P0  runs grok                    >=2/3 seeds per modulus, window median, never last row (C27)
  P1  PRIMARY: omega vs zdd at 128 Gini_add < 0.184 -> omega; > 0.434 -> zdd; between -> ambiguous
  P2a the new primes carry a clock |Gini_mult - 0.5665| <= 0.10 at 127 and 131
  P2b key-frequency count          descriptive; C4's count clause was withdrawn at 5 seeds
  P5  C8 at two fresh CRT moduli   permutation p < 0.01 in >=2/3 seeds at 123 and 91
  P6  O5 max_logit                 exploratory, never a claim: 3 primes cannot support one

P2c (neuron tuning) is `analyze_o4.py <dir>`, P3 (product-character ablation) is
`analyze_n4.py <dir>`, and P1/P4's W1 rescoring is `analyze_omega.py --confirm`. Those three
take a directory and were smoke-tested on this kernel's own artifacts before the push.

Why this file exists: `analyze_k04.py` hard-codes `results/k02_grid` / `results/k04_extended`
and takes no directory argument, so it cannot analyse k09.

Prime powers and primes are vacuous for the CRT-dual law by construction (one maximal prime
power => the predicted set is every frequency) and are never counted as P5 support.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sympy import factorint, isprime
from src.analysis.sparsity import gini, key_freqs_5x_median
from src.analysis.transforms import unit_index
from analyze_n7 import mult_amplitude   # canonical amplitude helper, self-checked
import test_crt_law as T

D = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results/k09_primes_zdd"
MODULI = [127, 131, 128, 123, 91]
PRIMES, CRT_TESTABLE, DISCRIMINATOR = [127, 131], [123, 91], 128
GROK = 0.99
N113_GINI_MULT = 0.5665          # k03 T4 5-seed tail mean, from the ledger
OMEGA_BAND_HI, ZDD_NBR_LO = 0.184, 0.434   # both fixed in the pre-registration


def load(d=D):
    """Every arm-B run in `d`, classified on the WINDOW MEDIAN and read BY COLUMN NAME."""
    out = []
    for f in sorted(glob.glob(f"{d}/WE_B_thesis_n*_s*.npz")):
        z = np.load(f, allow_pickle=True)
        if "hist_cols" not in z.files:
            continue
        cols = [str(c) for c in z["hist_cols"]]
        if "test_acc" not in cols:          # hist[-1][3] means different things across arms
            continue
        cfg = json.loads(json.loads(str(z["provenance"]))["config"])
        acc = z["hist"][:, cols.index("test_acc")]
        rec = dict(n=cfg["n"], seed=cfg["seed"], path=f,
                   acc=float(np.median(acc[-10:])), grokked=float(np.median(acc[-10:])) > GROK,
                   W=z["W_E"][:cfg["n"]].astype(float))
        if "max_logit" in cols:
            rec["max_logit"] = float(np.median(z["hist"][-10:, cols.index("max_logit")]))
        out.append(rec)
    return out


def g_add(r):
    W = r["W"] - r["W"].mean(0, keepdims=True)
    return gini(T.freq_energy(W, r["n"]))


def g_mult(r):
    """Discrete-log reordering then the DFT over phi(n). CYCLIC ONLY -- None otherwise,
    and the caller must SKIP LOUDLY rather than treat None as a value."""
    idx, orders, _ = unit_index(r["n"])
    if len(orders) != 1:
        return None
    U = np.array(sorted(idx, key=lambda x: idx[x]))
    Wu = r["W"][U] - r["W"][U].mean(0, keepdims=True)
    return gini(T.freq_energy(Wu, len(U)))


def main():
    runs = load()
    if not runs:
        print(f"NOT LANDED: no arm-B runs under {D}")
        return
    v = {}
    print("=" * 92)
    print(f"k09 -- experiments/PREREGISTER_k09_primes_zdd.md      {len(runs)} runs in {D}")
    print("=" * 92)

    # ---- P0 -------------------------------------------------------------------------
    print("\nP0  groks (window median of the final 10 samples, never the last row -- C27)")
    ok = {}
    for n in MODULI:
        rs = [r for r in runs if r["n"] == n]
        g = sum(r["grokked"] for r in rs)
        ok[n] = len(rs) > 0 and g >= 2 and g / len(rs) >= 2 / 3
        accs = " ".join(f"{r['acc']:.3f}" for r in sorted(rs, key=lambda x: x["seed"]))
        print(f"    n={n:<4} {g}/{len(rs)} grokked   [{accs}]" + ("" if ok[n] else "   <-- P0 NOT MET"))
    v["P0"] = "HELD" if all(ok.get(n) for n in MODULI) else "PARTIAL"

    # ---- P1 PRIMARY -----------------------------------------------------------------
    print(f"\nP1  PRIMARY -- omega vs zdd at n={DISCRIMINATOR} = 2^7 (omega=1, zdd=0.500)")
    rs = [r for r in runs if r["n"] == DISCRIMINATOR and r["grokked"]]
    if not rs:
        v["P1"] = "NOT ASSESSABLE (no grokked run)"
        print(f"    no grokked run at n={DISCRIMINATOR} -> {v['P1']}")
    else:
        vals = [g_add(r) for r in rs]
        m = float(np.mean(vals))
        v["P1"] = ("OMEGA" if m < OMEGA_BAND_HI else
                   "ZDD" if m > ZDD_NBR_LO else "AMBIGUOUS")
        print(f"    Gini_add = {m:.3f}  over {len(vals)} grokked seeds "
              f"[{' '.join(f'{x:.3f}' for x in vals)}]")
        print(f"    omega=1 band 0.017-{OMEGA_BAND_HI}  |  zdd~0.500 neighbourhood {ZDD_NBR_LO}-0.589")
        print(f"    -> {v['P1']}" + ("   !! the omega band structure is BROKEN; FINDINGS 5.1c-bis "
                                     "must be retracted" if v["P1"] == "ZDD" else ""))

    # ---- P2 -------------------------------------------------------------------------
    print(f"\nP2a the new primes carry the clock   (|Gini_mult - {N113_GINI_MULT}| <= 0.10)")
    held, unassessable = [], False
    for n in PRIMES:
        rs = [r for r in runs if r["n"] == n and r["grokked"]]
        gs = [g_mult(r) for r in rs]
        gs = [x for x in gs if x is not None]
        if not gs:
            # NOT ASSESSABLE must PROPAGATE. Collapsing it to NOT HELD reports a criterion
            # as failed when it was never scored -- a false negative indistinguishable from
            # a real failure, which is the family of bug this project keeps finding.
            print(f"    n={n:<4} NOT ASSESSABLE (no grokked run)")
            unassessable = True; continue
        m = float(np.mean(gs)); d = abs(m - N113_GINI_MULT)
        held.append(d <= 0.10)
        print(f"    n={n:<4} Gini_mult {m:.4f}   delta {d:.4f}   "
              f"{'HELD' if d <= 0.10 else 'NOT HELD'}   [{' '.join(f'{x:.3f}' for x in gs)}]")
    v["P2a"] = ("NOT ASSESSABLE" if unassessable and not held else
                "HELD" if held and all(held) else "NOT HELD")

    print("\nP2b key-frequency count   DESCRIPTIVE -- C4's count clause was WITHDRAWN at 5 seeds")
    for n in MODULI:
        for r in sorted([x for x in runs if x["n"] == n and x["grokked"]], key=lambda x: x["seed"]):
            idx, orders, _ = unit_index(n)
            if len(orders) != 1:
                print(f"    n={n:<4} s{r['seed']}  SKIPPED LOUDLY -- (Z/{n})* is not cyclic, "
                      f"no discrete log"); continue
            # key_freqs_5x_median takes per-frequency norms, not (W, n).
            # Bin convention: mult_amplitude drops DC, so its index 0 is frequency 1 and
            # the +1 is required before any of these is called a frequency.
            ks = [k + 1 for k in key_freqs_5x_median(mult_amplitude(r["W"], n))]
            print(f"    n={n:<4} s{r['seed']}  n_key = {len(ks)}   {ks}")
    v["P2b"] = "descriptive"

    # ---- P2c ------------------------------------------------------------------------
    # analyze_o4.py takes a directory but hard-codes MODULI = [113, 121, 125], so
    # `analyze_o4.py results/k09_primes_zdd` scores NOTHING here and prints
    # "0/0 grokked seeds ... NOT HELD" -- a false failure on an empty set. I ran that
    # command on the smoke, saw exit 0, and recorded it as verified without checking it
    # had scored k09's moduli. Its measure() helper IS reusable, so P2c is scored here.
    print("\nP2c neuron tuning in dlog coordinates   (>= 0.70; C31 reads 92-100% at n=113)")
    import analyze_o4 as O
    p2c = []
    for n in MODULI:
        for r in sorted([x for x in runs if x["n"] == n and x["grokked"]],
                        key=lambda x: x["seed"]):
            m = O.measure(r["path"], n)
            if "skip" in m:
                print(f"    n={n:<4} s{r['seed']}  {m['skip']}")
                continue
            p2c.append(m["tuned"] >= 0.70)
            print(f"    n={n:<4} s{r['seed']}  tuned {m['tuned']:.3f}  raw {m['tuned_raw']:.3f}  "
                  f"shuffled {m['tuned_shuf']:.3f}  keys {m['top_ks']}  "
                  f"emb-overlap {m['overlap']}/{len(m['emb_keys'])}  "
                  f"{'HELD' if m['tuned'] >= 0.70 else 'NOT HELD'}")
    v["P2c"] = ("NOT ASSESSABLE" if not p2c else
                "HELD" if all(p2c) else "NOT HELD")

    # ---- P5 -------------------------------------------------------------------------
    print("\nP5  C8 at the two fresh CRT-testable moduli (123, 91). Primes/prime powers VACUOUS.")
    p5, p5_unassessable = [], False
    for n in CRT_TESTABLE:
        rs = [r for r in runs if r["n"] == n and r["grokked"]]
        if not rs:
            print(f"    n={n:<4} NOT ASSESSABLE (no grokked run)")
            p5_unassessable = True; continue
        ps, es = [], []
        for r in rs:
            W = r["W"] - r["W"].mean(0, keepdims=True)
            e = T.freq_energy(W, n)
            # The -1 is required, and omitting it fails silently. `predicted()` returns real
            # frequencies (multiples of n/q, from d >= 1); `freq_energy` returns bins k = 1..n//2
            # at array positions 0..n//2-1, so position = frequency - 1, as in test_crt_law.main().
            # Unshifted, every predicted frequency lands one bin off the key set and reads as
            # depletion (enrichment 0.25-0.55, p ~ 1.0) on moduli where C8 holds at 3.3-17.4x.
            enr, p = T.permutation_test(e, [k - 1 for k in T.predicted(n)])[:2]
            ps.append(p); es.append(enr)
        good = sum(p < 0.01 for p in ps)
        # A modulus with FEWER THAN 2 grokked seeds cannot meet a ">=2/3 seeds" rule at
        # all. That is NOT ASSESSABLE (P0 failed there), not NOT HELD -- scoring it as a
        # failure reports C8 as broken at a modulus where it was never testable.
        if len(ps) < 2:
            p5_unassessable = True
            print(f"    n={n:<4} enrichment [{' '.join(f'{x:.2f}' for x in es)}]   "
                  f"p [{' '.join(f'{x:.4f}' for x in ps)}]   only {len(ps)} grokked seed"
                  f" -> NOT ASSESSABLE at >=2/3 (P0 failed here), reported not scored")
            continue
        p5.append(good >= 2 and good / len(ps) >= 2 / 3)
        print(f"    n={n:<4} enrichment [{' '.join(f'{x:.2f}' for x in es)}]   "
              f"p [{' '.join(f'{x:.4f}' for x in ps)}]   p<0.01 in {good}/{len(ps)}")
    v["P5"] = ("NOT ASSESSABLE" if p5_unassessable and not p5 else
               "HELD where assessable" if p5 and all(p5) and p5_unassessable else
               "HELD" if p5 and all(p5) else "NOT HELD")

    # ---- P6 -------------------------------------------------------------------------
    print("\nP6  O5 max_logit   EXPLORATORY, NOT A CLAIM -- 3 primes against a 3.8x seed spread")
    logged = [r for r in runs if "max_logit" in r]
    have = [r for r in logged if r["grokked"]]
    if not logged:
        print("    max_logit NOT LOGGED in these artifacts (k02-k08 dropped the column)")
    elif not have:
        # distinct from the line above: the column is present, nothing grokked to read it on
        print(f"    max_logit IS logged ({len(logged)} runs) but NO RUN GROKKED -- nothing to report")
    else:
        for n in MODULI:
            rs = [r for r in have if r["n"] == n]
            if rs:
                print(f"    n={n:<4} {'PRIME' if isprime(n) else str(dict(factorint(n))):<22} "
                      f"max_logit {np.mean([r['max_logit'] for r in rs]):8.2f}   ({len(rs)} seeds)")
    v["P6"] = "exploratory"

    print("\n" + "=" * 92)
    print("SUMMARY   (P1 is PRIMARY)")
    for k in ("P0", "P1", "P2a", "P2b", "P2c", "P5", "P6"):
        print(f"  {k:<4} {v.get(k, '-')}")
    print("\nStill to run:  analyze_n4.py <dir> (P3)   analyze_omega.py --confirm (P4)")
    print("NOT analyze_o4.py -- it hard-codes MODULI=[113,121,125] and scores nothing here.")
    print("Fill the Outcome of experiments/PREREGISTER_k09_primes_zdd.md; edit nothing above it.")
    return v


def _selfcheck():
    # the moduli are what the pre-registration says they are
    assert [len(factorint(n)) for n in MODULI] == [1, 1, 1, 2, 2]
    assert isprime(127) and isprime(131) and not isprime(128)
    assert factorint(128) == {2: 7}, "n=128 must be 2^7"
    # 128/123/91 are NON-cyclic -- g_mult must refuse them, not return a number
    for n in (128, 123, 91):
        assert len(unit_index(n)[1]) != 1, f"(Z/{n})* unexpectedly cyclic"
        assert g_mult(dict(n=n, W=np.zeros((n, 8)))) is None, \
            f"g_mult must return None at non-cyclic n={n}, not a value"
    for n in PRIMES:
        assert len(unit_index(n)[1]) == 1
    # the P1 decision bands must not overlap, or the criterion decides nothing
    assert OMEGA_BAND_HI < ZDD_NBR_LO, "P1's two bands overlap -- the rule is vacuous"
    # BIN CONVENTION, asserted rather than trusted: freq_energy's array position is
    # frequency-1, so the largest predicted frequency (n//2) must land on the LAST bin.
    for nn in (120, 123, 91, 165):
        pr = T.predicted(nn)
        assert max(pr) <= nn // 2, f"predicted({nn}) exceeds the Nyquist bin"
        assert max(k - 1 for k in pr) < nn // 2, \
            f"predicted({nn})-1 must index inside a length-{nn//2} spectrum"
    # a PLANTED signal on the predicted set must read ENRICHED through this exact path
    npl = 165
    pr = [k - 1 for k in T.predicted(npl)]
    e = np.full(npl // 2, 1.0); e[pr] = 20.0
    enr, pp = T.permutation_test(e, pr)[:2]
    assert enr > 5 and pp < 0.01, f"planted CRT signal must read enriched, got {enr:.2f}"
    # ...and the UNSHIFTED indexing must NOT read it, which is the bug this guards
    e2 = np.full(npl // 2, 1.0); e2[pr] = 20.0
    bad = T.permutation_test(e2, [k for k in T.predicted(npl) if k < npl // 2])
    assert bad[0] < enr, "the off-by-one indexing should read WEAKER than the correct one"
    # a planted pure additive frequency must read as SPARSE, and noise as not
    rng = np.random.default_rng(0)
    k, n = 7, 91
    x = np.arange(n)
    Wp = np.stack([np.cos(2 * np.pi * k * x / n), np.sin(2 * np.pi * k * x / n)], 1)
    assert g_add(dict(n=n, W=Wp)) > g_add(dict(n=n, W=rng.standard_normal((n, 2)))), \
        "a planted single frequency must be sparser than noise"
    print("analyze_k09 selfcheck PASS -- moduli, non-cyclic refusal, disjoint P1 bands, "
          "planted-frequency sanity")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else (_selfcheck(), main())
