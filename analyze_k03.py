"""k03 (N8) analysis -- implements experiments/PREREGISTER_k03_grid_acts.md exactly.

Every statistic here was committed before the run. Anything not here is exploratory and
must be labelled so.

  H1  depth grades block variance over and above block size   (the C20 re-test)
  H2  constant-zero blocks are NOT quiet                       (n=121 J_11xJ_11)
  H3  the grading is not additive in depth                     (O6, n=125, size-matched)
  H4  step 40k is converged                                    (the measurement audit)
  H5  Arm A participation ratio is flat at 40k                 (O1)
"""
import glob, json, os, sys
import numpy as np
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.analysis.jblocks import block_variance, composes_to_zero, j_classes, spearman
from src.analysis.transforms import character_transform, energy
from src.analysis.sparsity import gini, participation_ratio

RES = sys.argv[1] if len(sys.argv) > 1 else "results/k03_grid_acts"
SEEDS = [0, 1, 2, 3, 4]
DRIFT = 0.05                      # H4/H5 threshold, pre-registered
MIN_SEEDS = 4                     # "in >=4 of 5 seeds", pre-registered


def depth(n, d):
    """Nilpotency depth of J_d: the k with d = p^k for the relevant prime power, else the
    number of prime factors of d counted with multiplicity (0 for the units)."""
    k, x = 0, d
    for p in sorted({q for q in range(2, n + 1) if n % q == 0 and all(q % r for r in range(2, q))}):
        while x % p == 0:
            x //= p
            k += 1
    return k


def cells_budget(n):
    """Largest per-block sample size that keeps a useful number of blocks."""
    sizes = sorted(len(v) for v in j_classes(n).values())
    return max(4, min(16, sizes[-2] * sizes[-2] if len(sizes) > 1 else 16))


def wls_depth_coefficient(vb, n):
    """Fit var ~ b0 + b1*log2(cells) + b2*depth(a) + b3*depth(b); return b2 + b3.

    log2(cells) is entered as a regressor because block size is the dominant driver
    (rho = +0.85..+1.00) and is structurally coupled to depth via |J_d| = phi(n/d).
    """
    cls = j_classes(n)
    X, y = [], []
    for (a, b), (m, _) in vb.items():
        X.append([1.0, np.log2(len(cls[a]) * len(cls[b])), depth(n, a), depth(n, b)])
        y.append(m)
    X, y = np.array(X), np.array(y)
    if len(y) < 5 or np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[2] + beta[3])


def load(arm, n, seed):
    f = f"{RES}/WE_{arm}_n{n}_s{seed}.npz"
    return np.load(f, allow_pickle=False) if os.path.exists(f) else None


def spectral(W, n):
    """Gini (on AMPLITUDE) and PR (on ENERGY) in the multiplicative basis.

    An earlier version returned Gini on squared energy, which reads 0.902 where the
    amplitude protocol reads 0.539 on the same run; it was the source of C24's retired
    "Gini drifts <3%" clause (the corrected drift is 0.3-16%, FINDINGS 3.5).

    Protocol: Gini on amplitude sqrt(||s_k||^2+||c_k||^2), DC dropped, conjugates folded
    to one representative per pair. `energy()` is the wrong folding for a Gini: it sums
    each +-k pair but leaves the self-conjugate Nyquist bin single, so that one bin is
    scaled by 1 where every other is scaled by 2.

    PR is deliberately left on `energy()` and unchanged: energy is the correct convention
    for a participation ratio, and C24's VERIFIED PR numbers were measured on this exact
    path. Re-folding it would move them with no re-measurement to back the new values.

    `analyze_n7.mult_amplitude` is the canonical amplitude helper and is self-checked, but
    it is not a drop-in here: it centres on the unit rows only (`W[U].mean`) where this
    path centres over all n rows. Kept separate so the difference stays visible.
    """
    W = W[:n] - W[:n].mean(0, keepdims=True)
    S = character_transform(W, n)
    e = energy(S)                                  # PR only -- see docstring
    ek = (np.abs(S) ** 2).sum(-1)                  # energy per frequency, unfolded
    amp, seen = [], set()
    for i in np.ndindex(*ek.shape):
        c = tuple((-a) % sz for a, sz in zip(i, ek.shape))
        if i in seen or c in seen or not any(i):   # `not any(i)` is the DC multi-index
            continue
        seen.add(i); seen.add(c)
        amp.append(ek[i])
    return gini(np.sqrt(np.array(amp))), participation_ratio(e)


def h1_h2_h3(moduli=(121, 125)):
    print("\n=== H1  depth coefficient beta2+beta3 (negative = depth grades variance) ===")
    verdict = {}
    for n in moduli:
        betas = []
        for s in SEEDS:
            z = load("B_thesis", n, s)
            if z is None or "mlp_acts" not in z.files:
                continue
            vb = block_variance(z["mlp_acts"], n, cells=cells_budget(n))
            nblocks = len(vb)
            b = wls_depth_coefficient(vb, n)
            if b is not None:
                betas.append(b)
        if not betas:
            print(f"  n={n}: NO DATA"); continue
        betas = np.array(betas)
        neg = int((betas < 0).sum())
        if len(betas) < MIN_SEEDS:
            verdict[n] = None
            print(f"  n={n}: beta2+beta3 = {betas.mean():+.4f} over {len(betas)} seed(s) "
                  f"-> INSUFFICIENT SEEDS (need {MIN_SEEDS})")
            print(f"        per seed: {np.round(betas, 4).tolist()}")
            continue
        sd = betas.std(ddof=1)
        ok = neg >= MIN_SEEDS and betas.mean() < -2 * sd
        verdict[n] = ok
        print(f"  n={n}: beta2+beta3 = {betas.mean():+.4f} +/- {sd:.4f}  "
              f"negative in {neg}/{len(betas)} seeds   -> {'HELD' if ok else 'NOT HELD'}")
        print(f"        per seed: {np.round(betas, 4).tolist()}")
        print(f"        NOTE df: {len(betas)} seeds; n={n} supplies {nblocks} blocks "
              f"to a 4-parameter fit ({nblocks - 4} df) -- thin by construction.")
    vals = [v for v in verdict.values() if v is not None]
    print("  H1 (C20 re-test):",
          "INSUFFICIENT SEEDS -- verdict deferred" if not vals else
          ("HELD" if all(vals) else "NOT HELD -> C20 RETRACTED"))

    print("\n=== H2  n=121: is the constant-zero block J_11 x J_11 quiet? ===")
    hits = []
    for s in SEEDS:
        z = load("B_thesis", 121, s)
        if z is None or "mlp_acts" not in z.files:
            continue
        vb = block_variance(z["mlp_acts"], 121, cells=100)
        a, b = vb.get((11, 11)), vb.get((1, 121))
        if a and b:
            hits.append(a[0] > 2 * b[0])
            print(f"  seed {s}: J_11xJ_11 {a[0]:.3f}   J_1xJ_121 {b[0]:.3f}   "
                  f"ratio {a[0]/max(b[0],1e-9):.2f}x  (both compose to constant 0)")
    print(f"  H2: ratio > 2 in {sum(hits)}/{len(hits)} seeds -> "
          + ("INSUFFICIENT SEEDS -- verdict deferred" if len(hits) < MIN_SEEDS else
             "HELD -- constant-zero does NOT mean quiet" if sum(hits) >= MIN_SEEDS else "NOT HELD"))

    print("\n=== H3  O6: n=125 (J_1,J_25) vs (J_5,J_5), both exactly 400 cells ===")
    hits = []
    for s in SEEDS:
        z = load("B_thesis", 125, s)
        if z is None or "mlp_acts" not in z.files:
            continue
        vb = block_variance(z["mlp_acts"], 125, cells=400)
        a, b = vb.get((1, 25)), vb.get((5, 5))
        if a and b:
            hits.append(a[0] > b[0])
            print(f"  seed {s}: (J_1,J_25) {a[0]:.3f}   (J_5,J_5) {b[0]:.3f}   "
                  f"{'>' if a[0] > b[0] else '<='}")
    print(f"  H3: holds in {sum(hits)}/{len(hits)} seeds -> "
          + ("INSUFFICIENT SEEDS -- verdict deferred" if len(hits) < MIN_SEEDS else
             "HELD -- grading is not additive in depth" if sum(hits) >= MIN_SEEDS else "NOT HELD"))


def h4_h5():
    print("\n=== H4/H5  convergence audit: is step 40k converged? ===")
    print(f"  criterion: |m(final) - m(final-10k)| / m(final) < {DRIFT:.0%}\n")
    for arm, moduli in (("A_replication", [113]), ("B_thesis", [113, 121, 125, 119, 120, 165])):
        for n in moduli:
            rows = []
            for s in SEEDS:
                z = load(arm, n, s)
                if z is None or "we_traj" not in z.files:
                    continue
                tr, st = z["we_traj"], z["we_traj_steps"]
                i_end = len(st) - 1
                i_ref = int(np.argmin(np.abs(st - (st[-1] - 10_000))))
                if i_ref == i_end:
                    continue
                g_e, p_e = spectral(tr[i_end][:n].astype(float), n)
                g_r, p_r = spectral(tr[i_ref][:n].astype(float), n)
                rows.append((s, abs(g_e - g_r) / max(abs(g_e), 1e-12),
                             abs(p_e - p_r) / max(abs(p_e), 1e-12), g_e, p_e))
            if not rows:
                print(f"  {arm} n={n}: NO TRAJECTORY"); continue
            dg = np.array([r[1] for r in rows]); dp = np.array([r[2] for r in rows])
            ok = int(((dg < DRIFT) & (dp < DRIFT)).sum())
            print(f"  {arm} n={n}: Gini drift {dg.mean():.1%}  PR drift {dp.mean():.1%}  "
                  f"converged in {ok}/{len(rows)} seeds -> {'CONVERGED' if ok >= MIN_SEEDS or (arm=='A_replication' and ok==len(rows)) else 'STILL MOVING'}")
            for s, a, b, g, p in rows:
                print(f"        seed {s}: Gini_mult {g:.3f} (drift {a:.1%})  PR_mult {p:.2f} (drift {b:.1%})")


if __name__ == "__main__":
    have = sorted(glob.glob(f"{RES}/*.npz"))
    print(f"k03 analysis -- {len(have)} checkpoints in {RES}")
    if not have:
        sys.exit("no results yet; pull the kernel first")
    h1_h2_h3()
    h4_h5()
