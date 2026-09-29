"""k06 (N10) analysis -- implements experiments/PREREGISTER_k06_horizon.md exactly.

Criteria 1-5 were committed before this run existed. Anything not in that file is
exploratory and is labelled so.

  R1  the PR converges by 120k        final-10k drift < 5% in >=2/3 seeds, all 4 moduli
  R2  the 40k reading is still usable |PR(120k) - PR(40k)| / PR(40k) < 25% in >=2/3
  R3  Gini was right to be trusted    |dGini| < 0.05 in >=2/3
  R4  no late grok, no de-grok        every grok < 40k; acc stays >= 0.99 once reached
  #5  the 40k slice replicates k03    same seeds -- REPORT the difference, never assume 0

CONVENTION (protocol invariant 2, and it cost three sessions): Gini on AMPLITUDE, DC
dropped, sin/cos combined; participation ratio on normalised ENERGY p_k = |A_k|^2/sum.
Amplitude into a participation ratio gives 12.5 where the correct value is 4.3.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.analysis.sparsity import gini, participation_ratio
from src.analysis.transforms import unit_index
from src import provenance
import test_crt_law as T

D = sys.argv[1] if len(sys.argv) > 1 else "results/k06_horizon"
K03 = "results/k03_grid_acts"
MOD, SEEDS = [125, 119, 120, 165], [0, 1, 2]
FINAL_WINDOW = 10_000            # k03's H4 window, unchanged
ANCHOR = 40_000                  # the k03 horizon


def spectra(W, n):
    """(Gini_add, PR_add, Gini_mult, PR_mult) for one W_E. mult is None when non-cyclic."""
    Wc = W[:n].astype(float)
    Wc = Wc - Wc.mean(0, keepdims=True)
    fa = T.freq_energy(Wc, n)
    out = [gini(fa), participation_ratio(fa ** 2), None, None]
    idx, orders, _ = unit_index(n)
    if len(orders) == 1:
        Ud = np.array(sorted(idx, key=lambda x: idx[x]))
        Wu = W[Ud].astype(float)
        Wu = Wu - Wu.mean(0, keepdims=True)
        fm = T.freq_energy(Wu, len(Ud))
        out[2], out[3] = gini(fm), participation_ratio(fm ** 2)
    return out


def trajectory(path, n):
    """Per-snapshot spectra along we_traj. This run's whole output is a trajectory."""
    z = np.load(path, allow_pickle=False)
    if "we_traj" not in z.files:
        return None
    steps = z["we_traj_steps"]
    vals = [spectra(w, n) for w in z["we_traj"]]
    return steps, vals, z


def at_step(steps, vals, target):
    """The snapshot at or nearest below `target`."""
    i = int(np.searchsorted(steps, target, side="right") - 1)
    return max(i, 0)


def drift(a, b):
    return abs(b - a) / max(abs(a), 1e-30)


def main():
    p = f"{D}/horizon_summary.json"
    if not os.path.exists(p):
        sys.exit(f"NOT LANDED: {p}")
    rows = json.load(open(p))
    print("=" * 78)
    print("k06 -- the convergence horizon at the four moduli C24 flagged")
    print("=" * 78)
    print(f"\n{len(rows)} runs present of the 12 pre-registered\n")

    verdict = {k: [] for k in ("R1", "R2", "R3")}
    print("-" * 78)
    print("R1 / R2 / R3 -- drift over the final 10k, and 120k vs the 40k reading")
    print("-" * 78)
    for n in MOD:
        r1 = r2 = r3 = 0
        tot = 0
        print(f"\n  n={n}")
        for s in SEEDS:
            f = f"{D}/WE_B_thesis_n{n}_s{s}.npz"
            if not os.path.exists(f):
                print(f"    seed {s}: NOT SAVED"); continue
            tr = trajectory(f, n)
            if tr is None:
                print(f"    seed {s}: we_traj NOT SAVED -- cannot measure drift"); continue
            steps, vals, _ = tr
            tot += 1
            last = len(steps) - 1
            prev = at_step(steps, vals, steps[last] - FINAL_WINDOW)
            anc = at_step(steps, vals, ANCHOR)
            # PR is column 3 where the unit group is cyclic, else the additive PR (col 1)
            col = 3 if vals[last][3] is not None else 1
            gcol = 2 if vals[last][2] is not None else 0
            d_pr = drift(vals[prev][col], vals[last][col])
            d_anchor = drift(vals[anc][col], vals[last][col])
            d_gini = abs(vals[last][gcol] - vals[anc][gcol])
            r1 += d_pr < 0.05
            r2 += d_anchor < 0.25
            r3 += d_gini < 0.05
            basis = "mult" if col == 3 else "add (non-cyclic)"
            print(f"    seed {s}: PR_{basis} {vals[anc][col]:.2f} @40k -> "
                  f"{vals[last][col]:.2f} @{int(steps[last])//1000}k   "
                  f"final-10k drift {d_pr:5.1%}   vs-40k {d_anchor:5.1%}   "
                  f"dGini {d_gini:.3f}")
        if tot:
            verdict["R1"].append(r1 >= 2); verdict["R2"].append(r2 >= 2)
            verdict["R3"].append(r3 >= 2)
            print(f"    -> R1 {r1}/{tot}   R2 {r2}/{tot}   R3 {r3}/{tot}  (each needs >=2)")

    for k, what, consequence in (
            ("R1", "PR converges by 120k",
             "PR does NOT settle at composite moduli -- RETIRE it as a reported statistic\n"
             "       there rather than chase a longer horizon (pre-committed in the prereg)"),
            ("R2", "the 40k reading is usable",
             "every published PR at these moduli is WRONG, not merely imprecise,\n"
             "       and must be restated"),
            ("R3", "Gini was right to be trusted",
             "C6, C9 and the whole of FINDINGS section 3 were also measured before\n"
             "       convergence, and inherit C24")):
        ok = verdict[k] and all(verdict[k])
        print(f"\n  {k} ({what}): {'HELD' if ok else 'NOT HELD'}"
              + ("" if ok else f"\n    -> {consequence}"))

    print("\n" + "-" * 78)
    print("R4 / criterion 4 -- no grok after 40k, and no de-grok")
    print("-" * 78)
    late = [r for r in rows if r["grok_step"] is not None and r["grok_step"] >= ANCHOR]
    degrok = [r for r in rows if r["grok_step"] is not None and r["final_test_acc"] < 0.99]
    for r in rows:
        print(f"  n={r['n']:<4} seed {r['seed']}: grok {r['grok_step']}  "
              f"final acc {r['final_test_acc']:.4f}"
              + ("   <-- GROKKED AFTER 40k" if r in late else "")
              + ("   <-- DE-GROKKED" if r in degrok else ""))
    print(f"  -> R4 {'HELD' if not late and not degrok else 'NOT HELD'}")
    if late:
        print("     k02/k03/k04 grokking times at these moduli are RIGHT-CENSORED,")
        print("     not measured. That matters more than R1.")

    print("\n" + "-" * 78)
    print("criterion 5 -- the 40k slice against k03, same seeds, run for run")
    print("-" * 78)
    print("  (same seed != bitwise-identical trajectory on GPU; this reports the")
    print("   difference rather than asserting it is zero)")
    for n in MOD:
        for s in SEEDS:
            a = f"{D}/WE_B_thesis_n{n}_s{s}.npz"
            b = f"{K03}/WE_B_thesis_n{n}_s{s}.npz"
            if not (os.path.exists(a) and os.path.exists(b)):
                continue
            ta = trajectory(a, n)
            if ta is None:
                continue
            steps, vals, _ = ta
            i = at_step(steps, vals, ANCHOR)
            k3 = spectra(np.load(b, allow_pickle=False)["W_E"], n)
            col = 3 if vals[i][3] is not None else 1
            gcol = 2 if vals[i][2] is not None else 0
            print(f"  n={n:<4} s{s}: Gini k06@40k {vals[i][gcol]:.3f} vs k03 {k3[gcol]:.3f}"
                  f"   PR {vals[i][col]:.2f} vs {k3[col]:.2f}")

    print("\n" + "-" * 78)
    print("criterion 6 -- provenance")
    print("-" * 78)
    seen = {}
    for r in rows:
        f = f"{D}/WE_B_thesis_n{r['n']}_s{r['seed']}.npz"
        try:
            pr = provenance.read(f) or {}
        except Exception:
            pr = {}
        seen[(pr.get("git_sha", "unknown")[:7], pr.get("kaggle_account", "unknown"))] = \
            seen.get((pr.get("git_sha", "unknown")[:7],
                      pr.get("kaggle_account", "unknown")), 0) + 1
    for (sha, acct), k in sorted(seen.items()):
        flag = "" if sha not in ("unknown", "__GIT_", "") else "   <-- NOT REPRODUCIBLE"
        print(f"  git_sha {sha}  account {acct}  x{k}{flag}")


if __name__ == "__main__":
    main()
