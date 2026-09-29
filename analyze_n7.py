"""N7 analysis -- scores the criteria pre-registered in
experiments/PREREGISTER_n7_engine.md. Committed BEFORE the data existed.

  PYTHONPATH=. .venv/bin/python analyze_n7.py [results_dir]
  PYTHONPATH=. .venv/bin/python analyze_n7.py --selfcheck

E3 (causality, C22/C23) is NOT here -- it is `analyze_n4.py results/n7_engine`, which
already implements the 200-draw permutation test and the non-cyclic product-character
path. Reusing it is the point: nothing about N7 needed a line changed in it.
"""
import json, os, sys
import numpy as np
from sympy import factorint

from src.analysis.sparsity import gini, participation_ratio, key_freqs_5x_median
from src.analysis.transforms import (unit_index, character_transform, energy,
                                     random_orthogonal)
from src import provenance
import test_crt_law as T

D = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results/n7_engine"
MODULI, SEEDS = [113, 121, 125, 119], [0, 1, 2]
GINI_THRESH, TAIL = 0.10, 10_000
K03 = "results/k03_grid_acts"


def mult_amplitude(W, n):
    """Amplitude per multiplicative-character frequency: DC dropped, conjugates folded.

    Cyclic n: equals test_crt_law.freq_energy up to a constant factor, and Gini is
    scale-free -- asserted in _selfcheck. Non-cyclic n: the product-character analogue
    over the exponent tuple, which freq_energy cannot express because it assumes one
    cyclic group of order phi(n).

    Amplitude, not energy: squared energy inflates Gini 0.55 -> 0.90, and every Gini
    number this project reports is on amplitude.
    """
    idx, _, _ = unit_index(n)
    U = sorted(idx)
    W = np.asarray(W, dtype=float)
    S = character_transform(W - W[U].mean(0, keepdims=True), n)
    e = (np.abs(S) ** 2).sum(-1)               # energy per frequency, summed over features
    # ONE representative per conjugate pair, DC dropped -- freq_energy's convention.
    # energy() cannot be used here: it SUMS each pair (e[k]+e[-k]) but leaves the
    # self-conjugate Nyquist bin single, so that one bin ends up scaled by 1 where every
    # other is scaled by 2. Gini is scale-free but not scale-free per-bin, and the
    # selfcheck below catches exactly that (it did, on the first run).
    out, seen = [], set()
    for i in np.ndindex(*e.shape):
        c = tuple((-a) % sz for a, sz in zip(i, e.shape))
        if i in seen or c in seen or not any(i):
            continue                            # `not any(i)` is the DC multi-index
        seen.add(i); seen.add(c)
        out.append(e[i])
    return np.sqrt(np.array(out))


def add_amplitude(W, n):
    """Additive-basis amplitude, DC dropped -- the control basis for 'sparse in X'."""
    W = np.asarray(W, dtype=float)
    return T.freq_energy(W - W.mean(0, keepdims=True), n)


def rand_amplitude(W, n, seed=0):
    """Random orthogonal basis -- the other required control."""
    idx, _, _ = unit_index(n)
    U = sorted(idx)
    W = np.asarray(W, dtype=float)
    S = random_orthogonal(W[U] - W[U].mean(0, keepdims=True), seed=seed)
    e = np.sqrt(energy(S, fold_conjugate=False))
    # Dimension-matched: mult_amplitude folds phi(n) rows into phi(n)/2 bins, so compare
    # against the same count. Random orthogonal directions are exchangeable, so taking
    # the first half is unbiased -- and Gini does depend on component count.
    return e[:len(e) // 2]


def load(n, s, d=None):
    """One run, by (n, seed), whichever naming the producing code used.

    The engine writes `WE_engine_*`; the Kaggle kernels and `run_o18_cpu.py` write
    `WE_B_thesis_*`. Globbing only the first made `analyze_n9.py results/o18_cpu` -- the
    command O18's OWN pre-registration names as its analysis plan -- find zero runs and
    report "P0 NOT ASSESSABLE", which is indistinguishable from a sweep that failed to
    grok. Smoke-testing the scorer against the directory it will be pointed at costs
    seconds; `analyze_n7.py` crashed on every call once for the same class of reason.
    """
    for pat in (f"WE_engine_n{n}_s{s}.npz", f"WE_B_thesis_n{n}_s{s}.npz"):
        f = f"{d or D}/{pat}"
        if os.path.exists(f):
            return np.load(f, allow_pickle=False)
    return None


def tail_gini(z, n, window=TAIL):
    """E2's statistic: mean Gini_mult over the final `window` steps of the trajectory.

    A single-checkpoint read carries sd 0.02-0.044 (measured on k03 before this ran), so
    a single-point E2 would be a ~2 sd test against a 0.10 threshold. See the
    pre-registration's 'Estimator noise' section.
    """
    if "we_traj" not in z.files or len(z["we_traj"]) == 0:
        return None
    tr, st = z["we_traj"], z["we_traj_steps"]
    # The run's final step, not the trajectory's -- they differ when a run stopped between
    # `we_traj` samples. Only the engine's own npz carries `step`; k03's Kaggle-written
    # files have `we_traj_steps` but no `step`, and E5 pairs against them on EVERY call, so
    # reading it unconditionally crashed E5 (and, since main() evaluates e1..e5 in one
    # tuple, the whole analysis) no matter what data was present. For those files the last
    # trajectory sample is the best available estimate of where the run ended.
    end = int(z["step"]) if "step" in z.files else int(st[-1])
    sel = [i for i, t in enumerate(st) if t >= end - window]
    if not sel:
        return None                            # trajectory does not reach the tail; the
                                               # caller falls back and labels it (1pt)
    v = [gini(mult_amplitude(tr[i][:n].astype(float), n)) for i in sel]
    return float(np.mean(v)), float(np.std(v)), len(v), [int(st[i]) for i in sel]


def grokked(z):
    """E2/E4/E5 are claims about GROKKED circuits. The 150-step rehearsal passed E2 3/3
    and E4 3/3 on models at test acc 0.03 -- Gini was ~0.025 at every modulus, so
    comparing two near-zero numbers cleared the 0.10 threshold trivially, and CRT
    enrichment is already present within 150 steps. An ungrokked run is the untrained
    control, not a data point: it is excluded and said so, never silently counted."""
    return final_acc(z) > 0.99


def final_acc(z):
    h, c = z["hist"], [str(x) for x in z["hist_cols"]]
    return float(h[-1, c.index("test_acc")])


def e1(runs):
    print("\n" + "=" * 78 + "\nE1 -- does the engine grok at all four moduli?\n" + "=" * 78)
    print(f"{'n':>5} {'seeds present':>14} {'test acc per seed':>34} {'grok >=2/3':>18}")
    ok = {}
    for n in MODULI:
        accs = {s: final_acc(runs[(n, s)]) for s in SEEDS if (n, s) in runs}
        g = sum(a > 0.99 for a in accs.values())
        ok[n] = (f"HELD ({g}/{len(accs)})" if g >= 2 and accs else
                 (f"NOT ASSESSABLE (0 runs)" if not accs else f"NOT HELD ({g}/{len(accs)})"))
        print(f"{n:>5} {len(accs):>14} "
              f"{'  '.join(f's{s}:{a:.4f}' for s, a in accs.items()):>34} "
              f"{ok[n]:>18}")
    return ok


def e2(runs):
    print("\n" + "=" * 78)
    print("E2 -- C6: does the clock survive prime powers, on paper data?\n" + "=" * 78)
    print("Gini on AMPLITUDE, tail mean over the final 10k steps. Controls alongside.\n")
    print(f"{'run':>12} {'Gini_mult':>18} {'Gini_add':>9} {'Gini_rand':>10} "
          f"{'PR(energy)':>11} {'n_key':>6}")
    G = {}
    for n in MODULI:
        for s in SEEDS:
            z = runs.get((n, s))
            if z is None:
                continue
            t = tail_gini(z, n)
            W = z["W_E"][:n].astype(float)
            fm = mult_amplitude(W, n)
            if t:
                G[(n, s)] = t[0]
                gm = f"{t[0]:.3f}+-{t[1]:.3f}({t[2]})"
            else:
                gm = f"{gini(fm):.3f}(1pt)"
            print(f"{f'n={n} s={s}':>12} {gm:>18} {gini(add_amplitude(W, n)):>9.3f} "
                  f"{gini(rand_amplitude(W, n)):>10.3f} "
                  f"{participation_ratio(fm ** 2):>11.2f} "
                  f"{len(key_freqs_5x_median(fm)):>6}")
    print(f"\n{'comparison':>22} {'per-seed |delta|':>34} {'verdict':>16}")
    ok = {}
    for n in (121, 125):
        usable = [s for s in SEEDS if (n, s) in G and (113, s) in G
                  and grokked(runs[(n, s)]) and grokked(runs[(113, s)])]
        dropped = [s for s in SEEDS if (n, s) in G and (113, s) in G and s not in usable]
        ds = [abs(G[(n, s)] - G[(113, s)]) for s in usable]
        held = sum(d < GINI_THRESH for d in ds)
        v = (f"HELD ({held}/{len(ds)})" if len(ds) >= 2 and held >= 2 else
             ("NOT ASSESSABLE" if len(ds) < 2 else f"NOT HELD ({held}/{len(ds)})"))
        ok[n] = v
        print(f"{f'|Gini({n}) - Gini(113)|':>22} "
              f"{'  '.join(f'{d:.3f}' for d in ds) or '-':>34} {v:>16}")
        if dropped:
            print(f"{'':>22} excluded, did not grok: seeds {dropped}")
    print(f"\nthreshold {GINI_THRESH} -- the same one C6, k04 P4 and k05 Q2 use.")
    print("119 is reported above but is NOT part of E2: it is non-cyclic, so its "
          "Gini_mult is\nthe product-character analogue, not the same statistic.")
    return ok


def e4(runs):
    print("\n" + "=" * 78)
    print("E4 -- C8, the CRT-dual law. Only n=119 is testable here.\n" + "=" * 78)
    ok = {}
    for n in MODULI:
        if len(factorint(n)) == 1:
            print(f"n={n:<4} VACUOUS -- one CRT component, the predicted set is every "
                  f"frequency. Not support.")
            continue
        ps, dropped = [], []
        for s in SEEDS:
            z = runs.get((n, s))
            if z is None:
                continue
            if not grokked(z):
                dropped.append(s)
                continue
            W = z["W_E"][:n].astype(float)
            e = T.freq_energy(W - W.mean(0, keepdims=True), n)
            r = T.permutation_test(e, [k - 1 for k in T.predicted(n)])   # index = k-1
            if r:
                ps.append((s, r[0], r[1]))
        held = sum(p < 0.01 for _, _, p in ps)
        v = (f"HELD ({held}/{len(ps)})" if len(ps) >= 2 and held >= 2 else
             ("NOT ASSESSABLE" if len(ps) < 2 else f"NOT HELD ({held}/{len(ps)})"))
        ok[n] = v
        print(f"n={n:<4} " + ("  ".join(f"s{s}: enrich {o:.2f} p={p:.2e}" for s, o, p in ps)
                              or "no grokked seeds") + f"   {v}")
        if dropped:
            print(f"      excluded, did not grok: seeds {dropped}")
    return ok


def e5(runs):
    print("\n" + "=" * 78)
    print("E5 -- does the from-scratch engine agree with PyTorch? (no new runs)\n" + "=" * 78)
    print("Paired at the SAME seeds against k03. Band = 1 across-seed sd of the k03 value.\n")
    band = {113: 0.026, 121: 0.028, 125: 0.048}
    print(f"{'n':>5} {'engine':>16} {'k03 torch':>16} {'|delta|':>9} {'band':>7} {'verdict':>14}")
    ok = {}
    for n in (113, 121, 125):
        eng = [tail_gini(runs[(n, s)], n) for s in SEEDS
               if (n, s) in runs and grokked(runs[(n, s)])]
        eng = [t[0] for t in eng if t]
        tor = []
        for s in SEEDS:
            f = f"{K03}/WE_B_thesis_n{n}_s{s}.npz"
            if os.path.exists(f):
                t = tail_gini(np.load(f, allow_pickle=False), n)
                if t:
                    tor.append(t[0])
        if not eng or not tor:
            ok[n] = ("NOT ASSESSABLE (no grokked engine run)" if not eng
                     else "NOT ASSESSABLE (no k03 run to pair against)")
            print(f"{n:>5} {ok[n]:>16}")
            continue
        d = abs(np.mean(eng) - np.mean(tor))
        ok[n] = ("NOT ASSESSABLE" if len(eng) < 2 else
                 ("HELD" if d < band[n] else "NOT HELD"))
        print(f"{n:>5} {np.mean(eng):>10.3f}+-{np.std(eng):.3f} "
              f"{np.mean(tor):>10.3f}+-{np.std(tor):.3f} {d:>9.3f} {band[n]:>7.3f} "
              f"{ok[n]:>14}")
    print("\nC10 verifies the engine's gradients against PyTorch to 2.3e-15, so a "
          "systematic\ndisagreement here is a finding about the science, not an engine "
          "bug (prereg E5).")
    return ok


def _selfcheck():
    """The convention is the one thing that could silently invalidate every number, so
    assert the general path equals the tested one where both are defined."""
    n = 113
    rng = np.random.default_rng(0)
    W = rng.standard_normal((n, 16))
    idx, _, _ = unit_index(n)
    Ud = np.array(sorted(idx, key=lambda x: idx[x]))
    Wu = W[Ud] - W[Ud].mean(0, keepdims=True)
    a, b = gini(T.freq_energy(Wu, len(Ud))), gini(mult_amplitude(W, n))
    assert abs(a - b) < 1e-9, f"convention drift: freq_energy {a:.6f} vs general {b:.6f}"
    print(f"  cyclic Gini_mult: freq_energy {a:.6f} == general path {b:.6f}   OK")
    # The energy inflation appears on a sparse spectrum, not on noise (W above is random,
    # where both conventions read ~0.07), so test it on a clock-like spectrum with a few
    # dominant key frequencies.
    sp = np.full(56, 0.02); sp[[3, 16, 55]] = [1.0, 0.9, 0.8]
    g_a, g_e = gini(sp), gini(sp ** 2)
    assert g_e > g_a + 0.2, f"squaring failed to inflate Gini ({g_a:.3f} -> {g_e:.3f})"
    print(f"  clock-like spectrum: amplitude {g_a:.3f} vs energy {g_e:.3f} -- the "
          f"inflation is real, amplitude is the path in use   OK")
    f = "results/k03_grid_acts/WE_B_thesis_n113_s0.npz"
    if os.path.exists(f):
        Wr = np.load(f, allow_pickle=False)["W_E"][:113].astype(float)
        a_r = gini(mult_amplitude(Wr, 113))
        assert 0.40 < a_r < 0.70, f"trained Gini_mult {a_r:.3f} outside the reported band"
        print(f"  real trained checkpoint: Gini_mult {a_r:.3f} (C6 band 0.55 +- 0.05, and "
              f"energy would read ~0.90)   OK")
    else:
        print("  real-checkpoint cross-check: k03 NOT PRESENT -- skipped loudly")
    idx119, orders, _ = unit_index(119)
    assert orders == (6, 16) and len(mult_amplitude(np.zeros((119, 4)), 119)) > 0
    print(f"  n=119 product-character path defined, orders {orders}   OK")
    print("selfcheck PASS")


def main():
    if "--selfcheck" in sys.argv:
        return _selfcheck()
    runs = {(n, s): z for n in MODULI for s in SEEDS
            if (z := load(n, s)) is not None}
    if not runs:
        sys.exit(f"no WE_engine_*.npz in {D} -- refusing to print an empty table")
    print("=" * 78)
    print(f"N7 -- T2 on the FROM-SCRATCH ENGINE ({len(runs)} of "
          f"{len(MODULI) * len(SEEDS)} runs present)")
    print("=" * 78)
    for n in MODULI:
        for sd in SEEDS:
            if (n, sd) in runs:
                pr = provenance.read(f"{D}/WE_engine_n{n}_s{sd}.npz") or {}
                print(f"  code version: {pr.get('git_sha', '?')[:12]} "
                      f"dirty={pr.get('git_dirty')}  first run stamped "
                      f"{pr.get('timestamp_utc', '?')}")
                break
        else:
            continue
        break
    r1, r2, r4, r5 = e1(runs), e2(runs), e4(runs), e5(runs)
    print("\n" + "=" * 78 + "\nSUMMARY vs the pre-registration\n" + "=" * 78)
    for name, d in (("E1 groks", r1), ("E2 C6 clock", r2), ("E4 C8 CRT", r4),
                    ("E5 engine==torch", r5)):
        print(f"  {name:<20} " + ("  ".join(f"n={k}: {v}" for k, v in d.items())
                                    or "NOT ASSESSABLE (no runs present)"))
    print("\n  E3 causality  -> run: PYTHONPATH=. .venv/bin/python analyze_n4.py " + D)
    out = "results/n7_engine_summary.npz"
    np.savez(out, **provenance.stamp_npz(dict(source=D, n_runs=len(runs))),
             verdicts=np.array(json.dumps({k: {str(a): str(b) for a, b in v.items()}
                                           for k, v in (("E1", r1), ("E2", r2),
                                                        ("E4", r4), ("E5", r5))})))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
