"""J-class-blocked activation statistics, size-controlled.

The naive statistic -- variance over every (a,b) cell in a J_a x J_b block -- is
confounded: |J_d| = phi(n/d) shrinks as d grows, so deeper blocks hold fewer cells,
and rho(cell count, variance) is +0.85..+1.00 across all six moduli. A 1-cell block
(J_n x J_n, where J_n = {0}) reports variance exactly 0 because np.var of one sample
is 0 -- not because the network learned anything.

That is the same class-size confound that retracted C7b. Fix: measure every block at
the SAME number of cells by subsampling, and drop blocks that cannot supply that many.
See _selfcheck for the test that fails on the naive statistic and passes on this one.
"""
import numpy as np
from math import gcd


def j_classes(n):
    """{d: sorted members} for d | n, where J_d = {x in Z/nZ : gcd(x,n) = d}."""
    cls = {}
    for x in range(n):
        cls.setdefault(gcd(x, n), []).append(x)
    return {d: np.array(v) for d, v in sorted(cls.items())}


def block_variance(H, n, cells, n_boot=200, seed=0):
    """Per-(J_a,J_b) activation variance, every block measured at exactly `cells` cells.

    H: (n*n, n_neurons) or (n, n, n_neurons) post-ReLU MLP activations, grid-ordered.
    Returns {(a, b): (mean, sd)} over `n_boot` subsamples; blocks with fewer than
    `cells` cells are ABSENT from the result -- never reported as zero.
    """
    H = H.reshape(n, n, -1)
    cls = j_classes(n)
    rng = np.random.default_rng(seed)
    out = {}
    for a, ia in cls.items():
        for b, ib in cls.items():
            blk = H[np.ix_(ia, ib)].reshape(-1, H.shape[-1])
            if len(blk) < cells:
                continue
            v = [blk[rng.choice(len(blk), cells, replace=False)].var(0).mean()
                 for _ in range(n_boot)]
            out[(a, b)] = (float(np.mean(v)), float(np.std(v)))
    return out


def composes_to_zero(n, a, b):
    """True iff every product in the J_a x J_b block is 0 mod n."""
    cls = j_classes(n)
    return {(x * y) % n for x in cls[a] for y in cls[b]} == {0}


from src.analysis.stats import spearman   # midranks: block cell counts tie heavily


def naive_size_rho(H, n):
    """C20's retracted statistic: Spearman rho(block cell count, per-block var()).

    Every (J_a, J_b) block at its full size, the 1-cell block included. This is the
    statistic the size confound breaks; it is kept so the retraction's numbers recompute.
    """
    H = np.asarray(H).reshape(n, n, -1)
    cls = j_classes(n)
    sizes, var = [], []
    for ia in cls.values():
        for ib in cls.values():
            blk = H[np.ix_(ia, ib)].reshape(-1, H.shape[-1])
            sizes.append(len(blk))
            var.append(blk.var(0).mean())
    return spearman(sizes, var)


def _selfcheck():
    """Activations with block-INDEPENDENT variance must score flat under this statistic.

    The naive per-block var() reports a strong size gradient on exactly this input --
    which is the artifact that made C20 look like a depth law.
    """
    n = 125
    rng = np.random.default_rng(0)
    H = rng.normal(0, 1.0, size=(n * n, 8))          # same variance everywhere, by construction
    cls = j_classes(n)

    rho_naive = naive_size_rho(H, n)
    assert rho_naive > 0.6, f"naive statistic should show the size artifact, got {rho_naive}"
    one = H.reshape(n, n, -1)[np.ix_(cls[n], cls[n])].reshape(-1, 8)
    assert one.var(0).mean() == 0.0, "the 1-cell block must expose the np.var(1 sample)==0 artifact"

    fixed = block_variance(H, n, cells=16, n_boot=100)
    vals = np.array([m for m, _ in fixed.values()])
    assert (n, n) not in fixed, "1-cell block must be dropped, not reported as zero"
    spread = vals.max() - vals.min()
    assert spread < 0.25, f"size-controlled statistic must be flat on flat input, spread={spread:.3f}"
    print(f"  naive rho(size,var) = {rho_naive:+.3f} on flat input  <- the artifact")
    print(f"  size-controlled spread = {spread:.3f} over {len(vals)} blocks at 16 cells  <- flat")
    print("jblocks self-check PASS")


if __name__ == "__main__":
    _selfcheck()
