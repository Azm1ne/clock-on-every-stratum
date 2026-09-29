"""O4 -- neuron-level frequency purity, scoring the criteria pre-registered in
experiments/PREREGISTER_o4_neurons.md. Committed BEFORE the 121/125 numbers existed.

  PYTHONPATH=. .venv/bin/python analyze_o4.py [results_dir]
  PYTHONPATH=. .venv/bin/python analyze_o4.py --selfcheck

C6 -- the clock surviving at prime powers -- currently rests entirely on embedding spectra.
2606.17399's equivalent claim carries neuron-level support. This asks the MLP the same
question, on activations already saved by k03 and N7. No training.
"""
import os, sys
import numpy as np

from src.analysis.neurons import freq_fraction, shuffle_control
from src.analysis.runs import final_acc, state as run_state
from src.analysis.sparsity import key_freqs_5x_median
from src.viz.mechinterp import _grid, _dlog_order, _reorder_dlog
from src import provenance
import analyze_n7 as A

D = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results/k03_grid_acts"
MODULI, SEEDS = [113, 121, 125], [0, 1, 2, 3, 4]
THRESH = 0.85          # 2606.17399's and Nanda's threshold
TUNED_MIN = 0.60       # N1
RATIO_MIN = 10.0       # N2, N3
NKEY_MAX, COVER = 8, 0.50   # N4
DEAD = 1e-8            # a neuron whose grid variance is below this never fired


def path(d, n, s):
    for pat in (f"WE_B_thesis_n{n}_s{s}.npz", f"WE_engine_n{n}_s{s}.npz"):
        f = os.path.join(d, pat)
        if os.path.exists(f):
            return f
    return None


def measure(f, n, seed_for_shuffle=0):
    """All four numbers for one run. Returns None when the unit group is non-cyclic."""
    z = np.load(f, allow_pickle=False)
    if "mlp_acts" not in z.files:
        return dict(skip="no mlp_acts saved -- NOT SAVED is not absence of a phenomenon")
    order, orders = _dlog_order(n)
    if order is None:
        return dict(skip=f"unit group is non-cyclic {orders}; no discrete log. SKIPPED LOUDLY.")
    H = _grid(z["mlp_acts"], n).astype(float)
    Hd = _reorder_dlog(H, n, order)

    frac_d, bk_d = freq_fraction(Hd)
    frac_r, _ = freq_fraction(H)
    frac_s, _ = freq_fraction(shuffle_control(Hd, seed=seed_for_shuffle))

    tuned = frac_d > THRESH
    ks, counts = np.unique(bk_d[tuned], return_counts=True)
    top = np.sort(counts)[::-1][:NKEY_MAX].sum()
    cover = top / max(tuned.sum(), 1)

    W = z["W_E"][:n].astype(float)
    # +1 is correct here. `energy()` keeps DC at index 0, so its index is the frequency;
    # `mult_amplitude` drops DC, so its index 0 is frequency 1. Verified by planting a pure
    # character of known frequency k: mult_amplitude's argmax lands at k-1 (k=1->0, 5->4,
    # 17->16, 40->39) while freq_fraction's best_k lands at k. Comparing the two unshifted
    # makes every embedding key miss its neuron by one (overlap 0/4 for a set that matches).
    emb_keys = set(int(k) + 1 for k in key_freqs_5x_median(A.mult_amplitude(W, n)))
    top_ks = set(int(k) for k in ks[np.argsort(counts)[::-1][:NKEY_MAX]])

    h, c = z["hist"], [str(x) for x in z["hist_cols"]]
    return dict(
        s=Hd.shape[0],
        tuned=float(tuned.mean()),
        tuned_raw=float((frac_r > THRESH).mean()),
        tuned_shuf=float((frac_s > THRESH).mean()),
        mean_frac=float(frac_d.mean()),
        n_dead=int((Hd.var((0, 1)) < DEAD).sum()),
        n_distinct=int(len(ks)),
        cover=float(cover),
        top_ks=sorted(top_ks),
        emb_keys=sorted(emb_keys),
        overlap=len(top_ks & emb_keys),
        # Three states, not two, read from a window rather than the last row. k03's n125_s1
        # ends at 0.9779 (the figure C3 records from k02); calling that an ungrokked negative
        # control would put a model that has learned the circuit on the control side. A negative
        # control has to be a run that failed: the engine's n125_s1 (0.6082) and k04/k05's
        # 49/54/63 (0.15-0.28).
        # The window rule lives in src/analysis/runs.py for the whole project. Reading h[-1]
        # instead is the defect C27 was retracted for; over 162 runs the two readings disagree
        # on 4, none of them in k03.
        acc=final_acc(z),
        grokked=run_state(z)[0] == "grokked",
        state=run_state(z)[0],
    )


def _ratio(a, b):
    """tuned / control. The control is legitimately 0 here (both controls read 0.00% at
    n=113), and `inf` is the honest answer to 'how many times zero', not an error --
    but it is printed as `>=` a bound, never as a number to quote."""
    return np.inf if b <= 0 else a / b


def table(rows, title):
    print("\n" + "=" * 92 + f"\n{title}\n" + "=" * 92)
    print(f"{'run':>12} {'grid':>6} {'state':>9} {'tuned dlog':>11} {'raw':>8} {'shuffled':>9} "
          f"{'mean frac':>10} {'dead':>5} {'#freqs':>7} {'cover':>7} {'emb overlap':>12}")
    for name, r in rows:
        if "skip" in r:
            print(f"{name:>12}   {r['skip']}")
            continue
        print(f"{name:>12} {r['s']:>6} {r['state'][:9]:>9} "
              f"{100*r['tuned']:>10.1f}% {100*r['tuned_raw']:>7.1f}% {100*r['tuned_shuf']:>8.2f}% "
              f"{r['mean_frac']:>10.3f} {r['n_dead']:>5} {r['n_distinct']:>7} "
              f"{100*r['cover']:>6.0f}% {str(r['overlap'])+'/'+str(len(r['top_ks'])):>12}")


def n1(res):
    print("\n" + "=" * 92)
    print(f"N1 -- are the prime-power neurons single-frequency in multiplicative coordinates?")
    print("=" * 92)
    ok = {}
    for n in MODULI:
        rs = [r for (nn, s), r in res.items() if nn == n and "skip" not in r and r["grokked"]]
        hit = sum(r["tuned"] >= TUNED_MIN for r in rs)
        held = hit >= 4 and n in (121, 125)
        ok[n] = held
        note = "" if n in (121, 125) else "   (reference prime -- not a criterion)"
        print(f"  n={n}: {hit}/{len(rs)} grokked seeds at >= {TUNED_MIN:.0%} tuned"
              f"   {'HELD' if held else ('--' if n == 113 else 'NOT HELD')}{note}")
    return all(ok[n] for n in (121, 125))


def n2n3(res):
    print("\n" + "=" * 92)
    print("N2 / N3 -- is it the COORDINATE and the STRUCTURE, or the grid size?\n" + "=" * 92)
    print(f"{'run':>12} {'dlog/raw':>12} {'dlog/shuffled':>15} {'N2':>5} {'N3':>5}")
    ok2 = ok3 = True
    for (n, s), r in sorted(res.items()):
        if "skip" in r or not r["grokked"]:
            continue
        r2, r3 = _ratio(r["tuned"], r["tuned_raw"]), _ratio(r["tuned"], r["tuned_shuf"])
        g2, g3 = r2 >= RATIO_MIN, r3 >= RATIO_MIN
        ok2 &= g2; ok3 &= g3
        f = lambda x: ">= inf" if np.isinf(x) else f"{x:.1f}x"
        print(f"{f'n{n}_s{s}':>12} {f(r2):>12} {f(r3):>15} {'OK' if g2 else 'FAIL':>5} "
              f"{'OK' if g3 else 'FAIL':>5}")
    print(f"\n  N2 {'HELD' if ok2 else 'NOT HELD'}   N3 {'HELD' if ok3 else 'NOT HELD'}"
          f"   [both need >= {RATIO_MIN:.0f}x on every grokked run]")
    return ok2, ok3


def n4(res):
    print("\n" + "=" * 92)
    print("N4 -- do the tuned neurons concentrate on FEW frequencies? (the link back to C6)")
    print("=" * 92)
    print(f"{'run':>12} {'distinct k':>11} {'top-8 cover':>12} {'top-8 k':>26} "
          f"{'embedding keys':>26}")
    ok = True
    for (n, s), r in sorted(res.items()):
        if "skip" in r or not r["grokked"]:
            continue
        good = r["cover"] >= COVER
        if n in (121, 125):
            ok &= good
        print(f"{f'n{n}_s{s}':>12} {r['n_distinct']:>11} {100*r['cover']:>11.0f}% "
              f"{str(r['top_ks']):>26} {str(r['emb_keys']):>26}")
    print(f"\n  N4 {'HELD' if ok else 'NOT HELD'}   [<= {NKEY_MAX} distinct k cover "
          f">= {COVER:.0%} of tuned neurons, at 121 and 125]")
    return ok


def _selfcheck():
    from src.analysis.neurons import _selfcheck as ns
    ns()
    assert np.isinf(_ratio(0.9, 0.0)) and _ratio(0.9, 0.09) == 10.0
    # n=119 must be skipped, not silently scored: its unit group is Z6 x Z16
    f = path(D, 119, 0)
    if f:
        r = measure(f, 119)
        assert "skip" in r and "non-cyclic" in r["skip"], r
        print(f"  n=119 skipped loudly: {r['skip']}")
    print("analyze_o4: SELFCHECK PASS")


# The real negative control arm: runs that FAILED to generalise, with activations already
# on disk. Pooled from three experiments because no single one supplies enough of them.
CONTROL = [("results/n7_engine", "WE_engine_n125_s1.npz", 125),
           ("results/k04_extended", "WE_B_thesis_n49_s0.npz", 49),
           ("results/k04_extended", "WE_B_thesis_n49_s1.npz", 49),
           ("results/k04_extended", "WE_B_thesis_n54_s1.npz", 54),
           ("results/k04_extended", "WE_B_thesis_n63_s1.npz", 63),
           ("results/k04_extended", "WE_B_thesis_n63_s2.npz", 63)]


def control_arm():
    """Every sparsity claim in this project is corroborated and never contrasted. These are
    the matched failures -- same code, same hyperparameters, no generalisation.

    n=54 s0 (acc 0.7502) is NOT here: 54 = 2 * 3^3 has unit group Z_18, cyclic, but a run
    three quarters of the way to the answer is neither a grok nor a failure, and the same
    argument that excludes k03's 0.9779 near-grok excludes it. n=49's unit group is Z_42
    and n=63's is Z_6 x Z_6 -- non-cyclic, so 63 is skipped loudly by measure()."""
    rows = []
    for d, fn, n in CONTROL:
        f = os.path.join(d, fn)
        if not os.path.exists(f):
            continue
        rows.append((f"{os.path.basename(d)[:6]}/n{n}", measure(f, n)))
    print("\n" + "=" * 92)
    print("NEGATIVE CONTROL ARM -- runs that FAILED to generalise. Exploratory, no criterion.")
    print("=" * 92)
    if rows:
        table(rows, "failed runs (acc 0.15 - 0.61)")
    else:
        print("  none found on disk")


def main():
    res, rows = {}, []
    for n in MODULI:
        for s in SEEDS:
            f = path(D, n, s)
            if f is None:
                continue
            res[(n, s)] = measure(f, n)
            rows.append((f"n{n}_s{s}", res[(n, s)]))
    print(f"O4 -- neuron-level frequency purity.  data: {D}   threshold {THRESH}, 512 neurons")
    print(f"runs found: {len(rows)}")
    table(rows, "PER-RUN NUMBERS (dlog = multiplicative coordinates; raw and shuffled are the controls)")
    h1 = n1(res)
    h2, h3 = n2n3(res)
    h4 = n4(res)

    near = [(k, r) for k, r in res.items() if "skip" not in r and r.get("state") == "near-grok"]
    if near:
        print("\n" + "=" * 92)
        print("NEAR-GROKS -- excluded from the criteria, and NOT negative controls")
        print("=" * 92)
        print("  A run at 0.90 < acc <= 0.99 has learned the circuit; scoring it as a")
        print("  control would put a working model on the control side of the comparison.")
        table([(f"n{n}_s{s}", r) for (n, s), r in near], "near-groks")
    control_arm()

    print("\n" + "=" * 92 + "\nSUMMARY\n" + "=" * 92)
    for lab, v in (("N1 tuned >= 60% at 121 and 125", h1), ("N2 dlog vs raw", h2),
                   ("N3 dlog vs shuffled", h3), ("N4 few frequencies", h4)):
        print(f"  {lab:<34} {'HELD' if v else 'NOT HELD'}")
    print("\nFill the Outcome section of experiments/PREREGISTER_o4_neurons.md; never edit above it.")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
