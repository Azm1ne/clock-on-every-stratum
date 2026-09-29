"""Rank statistics and the Monte-Carlo p, in one place.

Two defects this module prevents.

1. ``np.argsort(np.argsort(x))`` is an ordinal rank, not Spearman's. It equals the
   Spearman rank only when no two values tie; with ties it breaks them by array
   position, so the statistic reads whatever the array happened to be ordered by.
   ``omega(n)`` takes three distinct values over 23 moduli, so the ordinal form ranks
   the nine ``omega = 1`` moduli in n-order: it gives ``rho(zdd, G | omega) = +0.735``
   above ``rho(omega, G | zdd) = +0.627``, where midranks give +0.578 and +0.804.

2. ``b / B`` is not a Monte-Carlo p-value: it reads exactly 0 whenever no draw is as
   extreme as the observation, claiming a resolution the design does not have. The
   paper's \\eqref{eq:permp} is the Phipson-Smyth ``(1 + b) / (B + 1)``.

The self-check plants a tie on purpose: on continuous data ties have probability zero,
so a check without one could not detect the first defect.
"""
import numpy as np


def rankdata(x):
    """Midranks, 1-based: tied values share the mean of the ranks they span.

    This is the rank Spearman's rho is defined on. `argsort(argsort(x))` is not.
    """
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y):
    """Spearman's rho: Pearson correlation of the midranks."""
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def partial_spearman(x, y, z):
    """rho(x, y | z) -- correlate the residuals of rank(x) and rank(y) on rank(z).

    Returns nan when a residual has zero variance (x fully determined by z), which
    is UNDEFINED rather than 1.0.
    """
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    resid = lambda r: r - np.polyval(np.polyfit(rz, r, 1), rz)
    ex, ey = resid(rx), resid(ry)
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def perm_p(n_ge, n_draws):
    """Phipson-Smyth (1 + b) / (B + 1). Never 0: the floor IS the resolution."""
    return (1.0 + float(n_ge)) / (float(n_draws) + 1.0)


def _selfcheck():
    # --- midranks, against the ordinal form that this module exists to retire ------
    ordinal = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    assert list(rankdata([10, 20, 30])) == [1.0, 2.0, 3.0]
    # a planted tie: the two 5s span ranks 2 and 3, so both read 2.5
    assert list(rankdata([1, 5, 5, 9])) == [1.0, 2.5, 2.5, 4.0], rankdata([1, 5, 5, 9])
    # ...and the ordinal form gets it wrong, in the direction that depends on ORDER
    assert list(ordinal([1, 5, 5, 9])) == [0.0, 1.0, 2.0, 3.0]

    # THE DEFECT, reproduced. A TWO-level grouping variable cannot perfectly rank
    # twelve distinct values -- rho = 1.0 is arithmetically unavailable to it. The
    # ordinal form reports exactly that, because within each level it ranks by
    # array position and the array happens to be sorted by y.
    g = [1] * 6 + [2] * 6
    y = list(range(12))
    r_ord = float(np.corrcoef(ordinal(g), ordinal(y))[0, 1])
    assert r_ord == 1.0, r_ord                      # spurious, and this is the bug
    assert spearman(g, y) < 0.90, spearman(g, y)    # the honest value
    # Spearman cannot depend on the order of tied entries; the ordinal form does.
    y2 = list(range(5, -1, -1)) + list(range(11, 5, -1))   # reverse within each level
    assert abs(spearman(g, y2) - spearman(g, y)) < 1e-12, "midrank must ignore within-tie order"
    # 1.000 -> 0.510 on a pure reordering of tied entries, against a midrank that
    # does not move at all. That swing IS the defect, in one line.
    assert float(np.corrcoef(ordinal(g), ordinal(y2))[0, 1]) < 0.60, \
        "the ordinal form must move on a reordering -- if this fires the demo is wrong"

    # --- partial correlation, on inputs whose answer is known by construction -------
    rs = np.random.default_rng(0)
    z, ea, eb = (rs.standard_normal(400) for _ in range(3))
    assert abs(partial_spearman(z + ea, z + eb, z)) < 0.15, "spurious partial correlation"
    assert partial_spearman(z + ea, z + ea + 0.05 * eb, z) > 0.90, "partial killed a real link"
    assert np.isnan(partial_spearman(z, z, z)), "zero-variance residual must be nan"

    # --- the p-value can never be 0, and is monotone in b --------------------------
    assert perm_p(0, 10_000) == 1 / 10_001
    assert perm_p(0, 200) > 0.0049 and perm_p(0, 20_000) < 5e-5
    assert perm_p(5, 100) > perm_p(4, 100)
    assert perm_p(100, 100) == 1.0, "every draw as extreme -> p = 1"
    print("src/analysis/stats selfcheck PASS -- midranks (tie planted), the ordinal form's "
          "order-sensitivity, partial correlation, and the (1+b)/(B+1) floor")


if __name__ == "__main__":
    _selfcheck()
