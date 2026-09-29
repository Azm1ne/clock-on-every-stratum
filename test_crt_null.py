"""C28-NULL -- is the early CRT enrichment CRT-specific, or just subgroup-shaped?

  PYTHONPATH=. .venv/bin/python test_crt_null.py [results_dir]
  PYTHONPATH=. .venv/bin/python test_crt_null.py --selfcheck

Criteria fixed in experiments/PREREGISTER_c28_crt_null.md, committed before any number
this script produces existed.

`test_crt_law.permutation_test` scores the CRT-dual set against uniformly random same-size
subsets. The predicted set is a union of two subgroups of Z/n -- 11 of 59 folded bins at
n=119 -- so a uniform null is weak against any alternative that concentrates energy on
arithmetic progressions through 0. That is the confound family behind the C7b, C19 and
C20 retractions, and the defence is to run the statistic on a control with the structure
but not the effect.

Three size-matched nulls:
  A  uniform random subsets                       -- the existing test, for comparison
  B  unions of multiples-of-d progressions,       -- primary: same shape, wrong divisors
     d ranging over non-divisors of n
  C  uniform subsets matched on mean frequency    -- isolates a low-|k| bias

Bin convention: freq_energy returns k = 1 .. n//2, DC dropped, so index i is frequency
i+1. _selfcheck plants a pure cosine of known frequency and asserts where it lands.
"""
import glob, os, sys
import numpy as np
from sympy import factorint

import test_crt_law as T
from src.analysis.transforms import unit_index   # noqa: F401  (parity with analyze_n7)
from src.analysis.stats import rankdata, perm_p

D = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results/n7_engine"
N_DRAW = 2_000
RNG = np.random.default_rng(0)
ALPHA = 0.01
# Overridable for O20, which scores the same ramp on a FAILED run at n=63. Pointed at
# another results dir WITHOUT this override the script finds no n=119 file and prints a
# confident report about an empty set -- the "running a command is not verifying it"
# failure. PREREGISTER_o20_failed_ramp.md criterion 4 re-runs the n=119 path to prove the
# override left the existing answer untouched.
PRIMARY_N = int(os.environ.get("CRT_NULL_N", 119))   # >=2 CRT components AND 1k resolution


def testable(n):
    """>=2 distinct prime powers. One component => n/q = 1 => every frequency predicted."""
    return len(factorint(n)) >= 2


def pred_idx(n):
    """Indices into freq_energy's output for the CRT-dual set. index = frequency - 1."""
    return np.array([k - 1 for k in T.predicted(n)])


def _progressions(n, m_bins):
    """Candidate 'multiples of d' index sets for d that does NOT divide n.

    Excluding divisors of n guarantees no draw can BE the true CRT set or a family of it,
    which is what makes B a null rather than a coin flip."""
    out = []
    for d in range(2, n // 2 + 1):
        if n % d == 0:
            continue
        s = np.arange(d, n // 2 + 1, d) - 1        # index = frequency - 1
        if 1 <= len(s) < m_bins:                   # a single family must not already fill it
            out.append(s)
    return out


def _draw_B(n, m, pool, rng):
    """A same-size union of 1..3 non-divisor progressions. Returns indices, or None."""
    for _ in range(40):
        u = np.array([], int)
        for _ in range(rng.integers(1, 4)):
            u = np.union1d(u, pool[rng.integers(len(pool))])
            if len(u) >= m:
                break
        if len(u) >= m:
            return rng.permutation(u)[:m]          # trim, never pad
    return None


def enrich(e, idx):
    mask = np.zeros(len(e), bool)
    mask[idx] = True
    return float(e[mask].mean() / e[~mask].mean())


def score(e, n, rng=None):
    """Enrichment of the CRT set, and p under each of the three nulls."""
    rng = rng or np.random.default_rng(0)
    m_all = len(e)
    idx = pred_idx(n)
    m = len(idx)
    if m == 0 or m >= m_all:
        return None
    obs = enrich(e, idx)
    ks = np.arange(1, m_all + 1)
    target_mean = ks[idx].mean()

    nulls = {k: [] for k in "ABC"}
    pool = _progressions(n, m_all)
    trimmed = 0
    for _ in range(N_DRAW):
        nulls["A"].append(enrich(e, rng.permutation(m_all)[:m]))
        b = _draw_B(n, m, pool, rng) if pool else None
        if b is not None:
            nulls["B"].append(enrich(e, b))
            trimmed += 1
        for _ in range(60):                         # rejection-sample the mean match
            c = rng.permutation(m_all)[:m]
            if abs(ks[c].mean() - target_mean) <= 2.0:
                nulls["C"].append(enrich(e, c))
                break
    out = dict(obs=obs, m=m, m_all=m_all, pred_mean=float(target_mean),
               B_draws=trimmed, pool=len(pool))
    for k, v in nulls.items():
        v = np.array(v)
        out[f"p_{k}"] = perm_p(int((v >= obs).sum()), len(v)) if len(v) else float("nan")
        out[f"null_{k}"] = float(v.mean()) if len(v) else float("nan")
    return out


def hist_acc(z):
    if "hist" not in z.files or "hist_cols" not in z.files:
        return None
    cols = [str(c) for c in z["hist_cols"]]
    return z["hist"][:, cols.index("test_acc")] if "test_acc" in cols else None


def sweep(d, n, seeds):
    """Criteria 1-4: enrichment vs step at one modulus, every we_traj snapshot."""
    rows = {}
    for s in seeds:
        f = next((p for p in (f"{d}/WE_engine_n{n}_s{s}.npz", f"{d}/WE_B_thesis_n{n}_s{s}.npz")
                  if os.path.exists(p)), None)
        if f is None:
            continue
        z = np.load(f, allow_pickle=True)
        if "we_traj" not in z.files:
            continue
        tr, ts = z["we_traj"], z["we_traj_steps"]
        acc = hist_acc(z)
        per = []
        for k in range(len(ts)):
            W = tr[k][:n].astype(float)
            e = T.freq_energy(W - W.mean(0, keepdims=True), n)
            r = score(e, n, np.random.default_rng(1000 + s))
            if r:
                r["step"] = int(ts[k])
                per.append(r)
        rows[s] = dict(rows=per, final_acc=float(acc[-1]) if acc is not None else float("nan"))
    return rows


def _ramp(per):
    """Spearman rho(step, enrichment) over 0..20k -- criterion 3."""
    w = [(r["step"], r["obs"]) for r in per if r["step"] <= 20_000]
    if len(w) < 4:
        return float("nan")
    return float(np.corrcoef(rankdata([x[0] for x in w]),
                             rankdata([x[1] for x in w]))[0, 1])


def discriminate():
    """Criterion 5 -- the G2 lesson. Pass RATE on grokked AND on FAILED runs, printed
    side by side. A statistic that passes on failures is not evidence, whatever it does
    on successes. Never thresholded here; the pair IS the output."""
    print("\n" + "=" * 92)
    print("CRITERION 5 -- DISCRIMINATION. The same statistic on grokked vs FAILED runs.")
    print("=" * 92)
    arms = {"grokked": [], "FAILED": [], "near-grok": []}
    for d in ("results/k03_grid_acts", "results/k04_extended", "results/k06_horizon"):
        for f in sorted(glob.glob(f"{d}/WE_*.npz")):
            base = os.path.basename(f)
            try:
                n = int(base.split("_n")[1].split("_")[0])
            except (IndexError, ValueError):
                continue
            if not testable(n):
                continue
            z = np.load(f, allow_pickle=True)
            acc = hist_acc(z)
            if acc is None:
                continue
            # Classify on a WINDOW, never the final row. C27 was retracted for exactly
            # this: a single last sample cannot tell a transient from a state, and k06's
            # n119_s0 ends inside a one-interval spike (FINDINGS 3.5b). Median of the
            # final 10 samples; `a_last` kept only to flag the disagreement out loud.
            a = float(np.median(acc[-10:]))
            a_last = float(acc[-1])
            arm = "grokked" if a > 0.99 else ("near-grok" if a > 0.90 else "FAILED")
            if abs(a - a_last) > 0.05:
                print(f"    [censoring] {base[3:-4]}: final row {a_last:.4f} but window "
                      f"median {a:.4f} -- classified on the window, as {arm}")
            W = z["W_E"][:n].astype(float)
            e = T.freq_energy(W - W.mean(0, keepdims=True), n)
            r = score(e, n, np.random.default_rng(7))
            if r:
                arms[arm].append((base[3:-4], n, a, r))
    for arm in ("grokked", "FAILED", "near-grok"):
        rs = arms[arm]
        if not rs:
            print(f"\n{arm}: none at a testable modulus")
            continue
        pa = sum(r["p_A"] < ALPHA for _, _, _, r in rs)
        nb_ok = [r for _, _, _, r in rs if not np.isnan(r["p_B"])]
        pb = sum(r["p_B"] < ALPHA for r in nb_ok)
        print(f"\n{arm}  (n={len(rs)})   null A passes {pa}/{len(rs)}   "
              f"null B passes {pb}/{len(nb_ok)} (B constructible in {len(nb_ok)}/{len(rs)})")
        for tag, n, a, r in rs:
            frac = r["m"] / r["m_all"]
            nb = "  n/a" if np.isnan(r["p_B"]) else f"{r['p_B']:.3f}"
            print(f"    {tag:24s} n={n:<4} acc {a:.4f}  set {r['m']:>2}/{r['m_all']:<2} "
                  f"({frac:.0%})  enrich {r['obs']:6.3f}  p_A {r['p_A']:.3f}  p_B {nb}"
                  + ("   <- near-VACUOUS set" if frac > 0.4 else ""))
    print("\n  A near-grok is neither arm -- it has learned the circuit, so scoring it as a")
    print("  control would put a working model on the control side (LAB_PROTOCOL.md).")
    return arms


def main():
    print("=" * 92)
    print(f"C28-NULL -- structure-matched null for the CRT-dual law    {D}")
    print("criteria: experiments/PREREGISTER_c28_crt_null.md")
    print("=" * 92)
    for n in (113, 121, 125):
        if not testable(n):
            print(f"n={n:<4} VACUOUS -- {factorint(n)} has one prime power, so n/q = 1 and the "
                  f"predicted set is EVERY frequency. Never counted as support.")

    res = sweep(D, PRIMARY_N, [0, 1, 2])
    if not res:
        print(f"\nno n={PRIMARY_N} runs with we_traj in {D}")
        return
    print(f"\nPRIMARY: n={PRIMARY_N} ({factorint(PRIMARY_N)}), {D}, all we_traj snapshots")
    r0 = next(iter(res.values()))["rows"][0]
    print(f"  predicted set {r0['m']} of {r0['m_all']} folded bins; mean |k| "
          f"{r0['pred_mean']:.1f} vs uniform {(r0['m_all'] + 1) / 2:.1f}; "
          f"null-B pool {r0['pool']} progressions\n")
    c1 = c2 = c3 = c4 = 0
    for s, v in res.items():
        print(f"  --- seed {s}   final acc {v['final_acc']:.4f} ---")
        print(f"  {'step':>7} {'enrich':>8} {'p_A':>8} {'p_B':>8} {'p_C':>8}")
        for r in v["rows"]:
            if r["step"] in (0, 1000, 2000, 5000, 10000, 20000, 40000):
                print(f"  {r['step']:7d} {r['obs']:8.3f} {r['p_A']:8.4f} "
                      f"{r['p_B']:8.4f} {r['p_C']:8.4f}")
        by = {r["step"]: r for r in v["rows"]}
        early = by.get(1000)
        late = by.get(max(by))
        init = by.get(0)
        rho = _ramp(v["rows"])
        c1 += bool(early and early["p_B"] < ALPHA)
        c2 += bool(late and late["p_B"] < ALPHA)
        c3 += rho > 0.8
        c4 += bool(init and init["p_B"] < ALPHA)      # criterion 4 wants this to stay 0
        print(f"  ramp rho(step, enrich) over 0-20k = {rho:.3f}")

    ns = len(res)
    print("\n" + "=" * 92 + "\nPRE-REGISTERED CRITERIA\n" + "=" * 92)
    for lab, got, need, ok in (
            ("1 PRIMARY  p_B < 0.01 at step 1,000", c1, f">=2/{ns}", c1 >= 2),
            ("2 p_B < 0.01 at the final snapshot", c2, f">=2/{ns}", c2 >= 2),
            ("3 ramp rho > 0.8 over 0-20k", c3, f">=2/{ns}", c3 >= 2),
            ("4 null CALIBRATION: step 0 must NOT reject", c4, f"==0/{ns}", c4 == 0)):
        print(f"  {lab:<44} {got}/{ns}  (need {need})  {'HELD' if ok else 'NOT HELD'}")
    discriminate()
    print("\nFill the Outcome section of experiments/PREREGISTER_c28_crt_null.md; "
          "never edit above it.")


def _selfcheck():
    n = 119
    m_all = n // 2
    # 1. Bin convention, settled by planting a signal.
    for f0 in (7, 17, 23):
        t = np.arange(n)
        W = np.cos(2 * np.pi * f0 * t / n)[:, None]
        e = T.freq_energy(W - W.mean(), n)
        assert int(np.argmax(e)) == f0 - 1, (f0, int(np.argmax(e)))
    # and the predicted-set indexer must agree with it
    assert pred_idx(119)[0] == 7 - 1
    assert set(T.predicted(119)) == {7,14,17,21,28,34,35,42,49,51,56}

    # 2. null B never contains the true CRT set, and is size-matched
    pool = _progressions(n, m_all)
    assert pool, "empty null-B pool"
    for s in pool:
        assert n % (s[0] + 1) != 0, "a DIVISOR of n leaked into the null-B pool"
    rng = np.random.default_rng(0)
    m = len(pred_idx(n))
    for _ in range(50):
        b = _draw_B(n, m, pool, rng)
        assert b is not None and len(b) == m and len(set(b.tolist())) == m

    # 3. the statistic is CALIBRATED: on white noise nothing rejects under any null
    e = np.abs(rng.normal(size=m_all)) + 1.0
    r = score(e, n, np.random.default_rng(3))
    assert 0.5 < r["obs"] < 2.0, r["obs"]
    assert min(r["p_A"], r["p_B"], r["p_C"]) > 0.01, r

    # 4. the statistic HAS POWER: planting energy on the CRT set rejects under all three
    e2 = np.ones(m_all)
    e2[pred_idx(n)] = 8.0
    r2 = score(e2, n, np.random.default_rng(4))
    assert r2["obs"] > 5 and max(r2["p_A"], r2["p_B"], r2["p_C"]) < 0.01, r2

    # 5. The point of this file: energy planted on a wrong progression must fool the
    #    uniform null and not fool the shape-matched one.
    e3 = np.ones(m_all)
    e3[np.arange(5, m_all + 1, 5) - 1] = 8.0        # multiples of 5; 5 does not divide 119
    r3 = score(e3, n, np.random.default_rng(5))
    assert r3["p_B"] > r3["p_A"], (r3["p_A"], r3["p_B"])

    # 6. vacuity gate
    assert not testable(121) and not testable(125) and not testable(113)
    assert testable(119) and testable(120) and testable(165)
    print("test_crt_null selfcheck PASS -- bin convention planted, null B divisor-free and "
          "size-matched, calibration, power, and B harder than A on a decoy progression")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
