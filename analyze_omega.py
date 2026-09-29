"""Additive sparsity is carried by omega(n), the count of distinct prime factors.

  PYTHONPATH=. .venv/bin/python analyze_omega.py [results_glob]
  PYTHONPATH=. .venv/bin/python analyze_omega.py --confirm   # W1-W6, PREREGISTER_omega.md
  PYTHONPATH=. .venv/bin/python analyze_omega.py --selfcheck

Exploratory, not pre-registered (the confirmatory W1-W6 are in PREREGISTER_omega.md).
Found while testing O7 (does cyclicity modulate C9?). It does not: cyclicity and omega are
collinear across our moduli, and where they disagree (n=54 = 2*3^3 and n=98 = 2*7^2, both
cyclic with omega=2) omega wins.

C9 records that additive sparsity tracks zero-divisor density (rho = +0.86, 18 moduli).
That correlation is not retracted and still holds within omega bands. What this adds is
that omega separates Gini_add into bands with no overlap, and that zdd is partly a proxy
for omega (rho(omega, zdd) = +0.77, midranks, 18 moduli).

Why omega is the mechanistically expected variable: C8's CRT-dual law predicts a frequency
set built as one family per maximal prime power. At omega = 1 that set is every frequency,
so a prime power has no additive structure to be sparse in, which is why its clock must
live in the multiplicative basis (C6). Each extra prime adds a family, hence more concentration, hence higher Gini_add.

Runs are classified by the median of the final 10 samples, never the last row (C27).
"""
import collections, glob, os, sys
import numpy as np
from sympy import factorint

from analyze_n7 import add_amplitude, mult_amplitude, rand_amplitude
from src.analysis.sparsity import gini
from src.analysis.transforms import unit_index
from src.analysis.stats import spearman, partial_spearman

PAT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results/k0*/WE_*.npz"
GROK = 0.99


# Spearman and the partial both come from src/analysis/stats.py, which ranks with
# MIDRANKS. omega takes three distinct values over 23 moduli, so the ordinal rank this
# file used to carry ranked the nine omega=1 moduli in n-order and inverted W2 vs W3.


def collect(pattern=PAT):
    """Per-modulus mean Gini_add over GROKKED runs only."""
    per = collections.defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        b = os.path.basename(f)[3:-4]
        try:
            n = int(b.split("_n")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        z = np.load(f, allow_pickle=True)
        if "hist_cols" not in z.files:
            continue
        cols = [str(c) for c in z["hist_cols"]]
        if "test_acc" not in cols:
            continue
        acc = z["hist"][:, cols.index("test_acc")]
        if float(np.median(acc[-10:])) <= GROK:      # window, never acc[-1] -- C27
            continue
        W = z["W_E"][:n].astype(float)
        per[n].append((gini(add_amplitude(W, n)), gini(rand_amplitude(W, n))))
    out = []
    for n, v in sorted(per.items()):
        ga = [x[0] for x in v]
        idx, orders, _ = unit_index(n)
        out.append(dict(n=n, omega=len(factorint(n)), zdd=1 - len(idx) / n,
                        cyclic=len(orders) == 1, ga=float(np.mean(ga)),
                        sd=float(np.std(ga)), rand=float(np.mean([x[1] for x in v])),
                        runs=len(v)))
    return out


def main():
    d = collect()
    if not d:
        print(f"no grokked runs under {PAT}")
        return
    print("=" * 92)
    print("ADDITIVE SPARSITY vs omega(n)   -- EXPLORATORY, not pre-registered")
    print("=" * 92)
    for w in sorted({x["omega"] for x in d}):
        g = [x for x in d if x["omega"] == w]
        lo, hi = min(x["ga"] for x in g), max(x["ga"] for x in g)
        print(f"\n  omega = {w}   {len(g)} moduli, {sum(x['runs'] for x in g)} runs   "
              f"Gini_add {lo:.3f} - {hi:.3f}   mean {np.mean([x['ga'] for x in g]):.3f}")
        for x in g:
            print(f"      n={x['n']:<4} {str(dict(factorint(x['n']))):<24} zdd {x['zdd']:.3f}  "
                  f"cyclic {str(x['cyclic']):<5} Gini_add {x['ga']:.3f}+-{x['sd']:.3f}  "
                  f"rand {x['rand']:.3f}  ({x['runs']} runs)")

    bands = {w: (min(x["ga"] for x in d if x["omega"] == w),
                 max(x["ga"] for x in d if x["omega"] == w))
             for w in sorted({x["omega"] for x in d})}
    ws = sorted(bands)
    overlap = any(bands[a][1] >= bands[b][0] for a, b in zip(ws, ws[1:]))
    print(f"\n  bands overlap: {overlap}"
          + ("" if overlap else "   <- omega separates Gini_add COMPLETELY"))

    W = [x["omega"] for x in d]; Z = [x["zdd"] for x in d]
    G = [x["ga"] for x in d];    C = [float(x["cyclic"]) for x in d]
    print(f"\n  rho(omega,  Gini_add) = {spearman(W, G):+.3f}")
    print(f"  rho(zdd,    Gini_add) = {spearman(Z, G):+.3f}   <- C9, not retracted")
    print(f"  rho(cyclic, Gini_add) = {spearman(C, G):+.3f}")
    print(f"  rho(omega,  zdd)      = {spearman(W, Z):+.3f}   <- why zdd looked causal")
    print(f"  rho(omega,  cyclic)   = {spearman(W, C):+.3f}   <- why cyclicity looked causal")

    print("\n  CONFOUND CHECK -- does zdd still predict WITHIN an omega band?")
    for w in ws:
        g = [x for x in d if x["omega"] == w]
        if len(g) >= 4:
            print(f"    omega={w} ({len(g)} moduli): rho(zdd, Gini_add) = "
                  f"{spearman([x['zdd'] for x in g], [x['ga'] for x in g]):+.3f}")
    print("\n  THE DECIDING CASES -- cyclic AND omega=2. Cyclicity predicts LOW, omega HIGH.")
    top1 = max((x["ga"] for x in d if x["omega"] == 1), default=float("nan"))
    for x in d:
        if x["cyclic"] and x["omega"] == 2:
            print(f"    n={x['n']:<4} {str(dict(factorint(x['n']))):<16} cyclic, omega=2  "
                  f"Gini_add {x['ga']:.3f}   (the omega=1 band tops out at {top1:.3f})")
    print("\n  O7 is resolved by REPLACEMENT: its premise (cyclicity modulates C9) does not")
    print("  survive, and omega(n) -- which C8's CRT-dual law already implies -- does.")
    print("  Still EXPLORATORY: it needs a pre-registration before it can be a claim.")


# ---------------------------------------------------------------------------------------
# Confirmatory tests W1-W6.  experiments/PREREGISTER_omega.md, committed 60ee31b BEFORE
# any statistic below was computed.  These do NOT make omega a pre-registered discovery --
# the bands were seen first.  They kill the confounds that killed C7b, C19 and C20.
# ---------------------------------------------------------------------------------------

def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial p, no scipy dependency."""
    from math import comb
    pmf = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return float(sum(x for x in pmf if x <= pmf[k] + 1e-12))


def band_ranges(d, key):
    return {w: (min(x[key] for x in d if x["omega"] == w),
                max(x[key] for x in d if x["omega"] == w))
            for w in sorted({x["omega"] for x in d})}


def min_adjacent_gap(omegas, vals):
    """Smallest gap between adjacent omega bands; negative if any adjacent pair overlaps."""
    ws = sorted(set(omegas))
    lo = {w: min(v for o, v in zip(omegas, vals) if o == w) for w in ws}
    hi = {w: max(v for o, v in zip(omegas, vals) if o == w) for w in ws}
    return min(lo[b] - hi[a] for a, b in zip(ws, ws[1:]))


def confirm(zdd_tol=0.05, draws=10_000, seed=0):
    d = collect()
    if not d:
        print(f"no grokked runs under {PAT}")
        return
    N = [x["n"] for x in d]; W = [x["omega"] for x in d]
    Z = [x["zdd"] for x in d]; G = [x["ga"] for x in d]
    v = {}
    print("=" * 92)
    print("W1-W6 CONFIRMATORY -- experiments/PREREGISTER_omega.md (committed before scoring)")
    print(f"  {len(d)} moduli, {sum(x['runs'] for x in d)} grokked runs, per-modulus means")
    print("=" * 92)

    # ---- W1 PRIMARY: does omega separate at MATCHED zero-divisor density? --------------
    pairs = [(a, b) for i, a in enumerate(d) for b in d[i + 1:]
             if abs(a["zdd"] - b["zdd"]) <= zdd_tol and a["omega"] != b["omega"]]
    print(f"\nW1  PRIMARY -- omega at matched zdd (|dzdd| <= {zdd_tol}) -- {len(pairs)} discordant pairs")
    if len(pairs) < 6:
        v["W1"] = "NOT ASSESSABLE"
        print("    fewer than 6 pairs -> NOT ASSESSABLE (a pre-registered outcome, not a skip)")
    else:
        agree = 0
        for a, b in pairs:
            hi, lo = (a, b) if a["omega"] > b["omega"] else (b, a)
            ok = hi["ga"] > lo["ga"]
            agree += ok
            print(f"    n={a['n']:<4}(w{a['omega']} zdd {a['zdd']:.3f} G {a['ga']:.3f})  vs  "
                  f"n={b['n']:<4}(w{b['omega']} zdd {b['zdd']:.3f} G {b['ga']:.3f})   "
                  f"{'higher-omega HIGHER' if ok else 'higher-omega LOWER  <-- against'}")
        frac = agree / len(pairs)
        pv = binom_two_sided(agree, len(pairs))
        v["W1"] = ("HELD" if frac >= 0.80 and pv < 0.05 else
                   "NOT HELD" if frac <= 0.60 else "PARTIAL")
        print(f"    fraction favouring omega = {agree}/{len(pairs)} = {frac:.3f}   "
              f"sign test (exact, two-sided) p = {pv:.2e}   -> {v['W1']}")
        print("    !! the pairs SHARE moduli and are NOT INDEPENDENT, so the sign-test p is")
        print("       optimistic and is descriptive only. The FRACTION is the statistic.")

    # ---- W2 / W3: partial correlations --------------------------------------------------
    w2 = partial_spearman(W, G, Z)
    w3 = partial_spearman(Z, G, W)
    v["W2"] = "HELD" if w2 >= 0.50 else "NOT HELD"
    v["W3"] = "descriptive"
    print(f"\nW2  rho(omega, Gini_add | zdd) = {w2:+.3f}   (raw {spearman(W, G):+.3f})  -> {v['W2']}")
    print(f"W3  rho(zdd, Gini_add | omega) = {w3:+.3f}   (raw {spearman(Z, G):+.3f})  -> descriptive")

    # ---- W4 SIZE CONTROL (NOT blind -- the control values were already visible) ----------
    rb = band_ranges(d, "rand")
    ws = sorted(rb)
    ov = any(rb[a][1] >= rb[b][0] for a, b in zip(ws, ws[1:]))
    v["W4"] = "HELD" if ov else "NOT HELD"
    print(f"\nW4  SIZE CONTROL -- residue-axis random-orthogonal bands   [NOT BLIND]")
    for w in ws:
        print(f"    omega={w}   rand Gini {rb[w][0]:.3f} - {rb[w][1]:.3f}")
    print(f"    control bands overlap: {ov}  -> {v['W4']}"
          + ("   (no band structure in the control)" if ov else "   <-- THE CONTROL SEPARATES TOO"))

    # ---- W5 SIZE CONTROL, blind ---------------------------------------------------------
    rng_, rnw = spearman(N, G), spearman(N, W)
    v["W5"] = "HELD" if abs(rng_) <= 0.40 else "NOT HELD"
    print(f"\nW5  SIZE CONTROL -- n itself   [BLIND]")
    print(f"    rho(n, Gini_add) = {rng_:+.3f}   rho(n, omega) = {rnw:+.3f}   -> {v['W5']}")

    # ---- W6 permutation on the band separation ------------------------------------------
    obs = min_adjacent_gap(W, G)
    rng = np.random.default_rng(seed)
    Wa = np.array(W)
    ge = sum(min_adjacent_gap(rng.permutation(Wa), G) >= obs for _ in range(draws))
    pv6 = (ge + 1) / (draws + 1)
    v["W6"] = "HELD" if pv6 < 0.01 else "NOT HELD"
    print(f"\nW6  permutation -- {draws} shuffles of the omega labels")
    print(f"    observed min adjacent band gap = {obs:+.3f}   p = {pv6:.2e}  -> {v['W6']}")

    # ---- pre-registered robustness: the omega=1 band without the project's one prime -----
    g1 = [x for x in d if x["omega"] == 1]
    nop = [x for x in g1 if x["n"] != 113]
    lo2 = min(x["ga"] for x in d if x["omega"] == 2)
    print(f"\nROBUSTNESS (pre-registered) -- the omega=1 band without the ONE prime")
    print(f"    with n=113   : {min(x['ga'] for x in g1):.3f} - {max(x['ga'] for x in g1):.3f}   ({len(g1)} moduli)")
    print(f"    without n=113: {min(x['ga'] for x in nop):.3f} - {max(x['ga'] for x in nop):.3f}   ({len(nop)} moduli)")
    print(f"    omega=2 band starts at {lo2:.3f}  -> still separated: {max(x['ga'] for x in nop) < lo2}")

    print("\n" + "=" * 92)
    print("SUMMARY   (W1 is PRIMARY)")
    for k in ("W1", "W2", "W3", "W4", "W5", "W6"):
        print(f"  {k}  {v[k]}")
    print("\nFill the Outcome section of experiments/PREREGISTER_omega.md; edit nothing above it.")
    print("This is a CONFIRMATORY test of an EXPLORATORY observation, on the same data.")
    print("It kills confounds. It does NOT make omega a pre-registered discovery.")
    return v


def _selfcheck():
    # omega is what it says it is, on the moduli this project actually uses
    assert len(factorint(113)) == 1 and len(factorint(125)) == 1   # prime, prime power
    assert len(factorint(54)) == 2 and len(factorint(98)) == 2     # the deciding cases
    assert len(factorint(165)) == 3
    # the deciding cases must really be CYCLIC, or the argument evaporates
    for n in (54, 98):
        assert len(unit_index(n)[1]) == 1, f"(Z/{n})* is not cyclic -- the argument breaks"
    # ...and the omega=1 comparison set must be cyclic too
    for n in (49, 81, 113, 121, 125, 169):
        assert len(unit_index(n)[1]) == 1
    # spearman sanity: monotone -> +1, reversed -> -1
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-12
    # A rotation of the feature axis is a no-op for these statistics (they sum energy over
    # features), so the control must rotate the residue axis.
    rng = np.random.default_rng(0)
    Wm = rng.standard_normal((49, 128))
    Q, _ = np.linalg.qr(rng.standard_normal((128, 128)))
    assert np.allclose(gini(mult_amplitude(Wm, 49)), gini(mult_amplitude(Wm @ Q, 49))), \
        "feature-axis rotation should be a no-op"
    assert not np.allclose(gini(mult_amplitude(Wm, 49)), gini(rand_amplitude(Wm, 49))), \
        "residue-axis control must actually change the spectrum"
    # the confirmatory helpers, on inputs whose answer is known by construction.
    # NOTE: x fully determined by z leaves a ZERO-variance residual, so the partial is
    # UNDEFINED, not 1.0 -- the first version of this check asserted 1.0 and was wrong.
    rs = np.random.default_rng(0)
    zc, ea, eb = (rs.standard_normal(400) for _ in range(3))
    # shared dependence on z only -> the partial must vanish
    assert abs(partial_spearman(zc + ea, zc + eb, zc)) < 0.15, "spurious partial correlation"
    # a genuine link beyond z -> the partial must survive
    assert partial_spearman(zc + ea, zc + ea + 0.05 * eb, zc) > 0.90, "partial killed a real link"
    assert np.isnan(partial_spearman(zc, zc, zc)), "zero-variance residual must report nan"
    assert abs(binom_two_sided(5, 5) - 2 * 0.5 ** 5) < 1e-12
    assert abs(binom_two_sided(0, 1) - 1.0) < 1e-12
    # min_adjacent_gap: separated bands -> positive, overlapping -> negative
    assert min_adjacent_gap([1, 1, 2, 2], [0.1, 0.2, 0.5, 0.6]) > 0
    assert min_adjacent_gap([1, 1, 2, 2], [0.1, 0.6, 0.5, 0.7]) < 0
    print("analyze_omega selfcheck PASS -- omega counts, the deciding cases are cyclic, "
          "spearman, and the feature-vs-residue rotation trap")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    elif "--confirm" in sys.argv:
        _selfcheck(); confirm()
    else:
        main()
