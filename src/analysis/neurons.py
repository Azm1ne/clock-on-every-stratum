"""Per-neuron frequency tuning: does ONE frequency explain a neuron's activation grid?

This is Nanda's Fig 5 statistic, and 2606.17399's Table 1 claim, made computable in both
bases. `src/viz/mechinterp.neuron_frequency_map` draws it; this module IS it, so the
picture and the number can never disagree.

The 8-bin family: within frequency k, a product of one-dimensional harmonics in a and b
lands only in the bins (+-k, 0), (0, +-k), (+-k, +-k). Summing those eight and dividing by
the neuron's total (DC-free) energy asks "is this neuron a single-frequency object?".
2606.17399 sums the same set as a 3x3 block {0, 2k-1, 2k}^2 in the REAL Fourier basis with
(0,0) zeroed -- the same eight numbers in a different bookkeeping.
"""
import numpy as np


def freq_fraction(H):
    """H: (s, s, n_neurons) real activation grid -> (frac, best_k) per neuron.

    frac[j] is the largest share of neuron j's DC-free 2D spectral energy that any single
    frequency's 8-bin family accounts for; best_k[j] is that frequency. A neuron with no
    energy at all returns frac 0, not a division by zero.
    """
    H = np.asarray(H, dtype=float)
    s = H.shape[0]
    assert H.shape[1] == s, f"grid is {H.shape[:2]}, not square"
    F = np.abs(np.fft.fft2(H - H.mean((0, 1), keepdims=True), axes=(0, 1))) ** 2
    tot = F.sum((0, 1))
    best_f = np.zeros(F.shape[-1])
    best_k = np.zeros(F.shape[-1], int)
    for k in range(1, s // 2 + 1):
        p, q = k % s, (-k) % s
        e = sum(F[i, j] for i, j in {(p, 0), (q, 0), (0, p), (0, q),
                                     (p, p), (q, q), (p, q), (q, p)})
        upd = e > best_f
        best_f[upd], best_k[upd] = e[upd], k
    return best_f / np.maximum(tot, 1e-30), best_k


def freq_fraction_nd(H, orders):
    """Product-character version of `freq_fraction`, for a NON-CYCLIC index group.

    `freq_fraction` is a 2-D cyclic statistic, so C31 had to skip n=119, 120 and 165 -- the
    moduli whose stratification is most interesting. (Z/nZ)* is abelian but not always
    cyclic; a character is then a multi-index kappa, and the eight-bin family of a single
    character is the same eight points written multi-index-wise:

        (+-kappa, 0) · (0, +-kappa) · (+-kappa, +-kappa)

    because multiplication of units is still ADDITION of exponent tuples -- the same reason
    `ablation.key_freq_pairs` puts f(a+b) on the diagonal for a product character. The
    cyclic case is r = 1 of this, not a separate path.

    H is `orders + orders + (n_neurons,)`. Returns (frac, best_kappa) per neuron.
    """
    r = len(orders)
    assert H.shape[:2 * r] == tuple(orders) + tuple(orders), (H.shape, orders)
    H = np.asarray(H, dtype=float)
    ax = tuple(range(2 * r))
    F = np.abs(np.fft.fftn(H - H.mean(ax, keepdims=True), axes=ax)) ** 2
    tot = F.sum(ax)
    zero = (0,) * r
    neg = lambda k: tuple((-x) % o for x, o in zip(k, orders))
    best_f = np.zeros(F.shape[-1])
    best_k = [None] * F.shape[-1]
    seen = set()
    for k in np.ndindex(*orders):
        if k == zero or k in seen:
            continue
        nk = neg(k)
        seen.add(k); seen.add(nk)
        fam = {(k, zero), (nk, zero), (zero, k), (zero, nk),
               (k, k), (nk, nk), (k, nk), (nk, k)}
        e = sum(F[a + b] for a, b in fam)
        upd = e > best_f
        best_f[upd] = e[upd]
        for j in np.nonzero(upd)[0]:
            best_k[j] = k
    return best_f / np.maximum(tot, 1e-30), best_k


def shuffle_control(H, seed=0):
    """The same grid with each neuron's cells permuted independently.

    The chance level of `freq_fraction` depends on the grid side (eight bins out of s^2,
    maximised over s/2 candidate frequencies), so a bare percentage cannot be compared
    across moduli whose unit groups have different orders. This control has the size
    structure and the activation distribution but no 2D structure: the size-confound
    control that C7b, C19 and C20 lacked.
    """
    rng = np.random.default_rng(seed)
    H = np.asarray(H, dtype=float)
    flat = H.reshape(-1, H.shape[-1]).copy()
    for j in range(flat.shape[1]):
        flat[:, j] = rng.permutation(flat[:, j])
    return flat.reshape(H.shape)


def tuned_fraction(H, thresh=0.85):
    """Share of neurons a single frequency explains above `thresh`. 2606.17399 reports
    96.9% at this threshold for a*b mod p; Nanda reports 84.6% for a+b."""
    frac, _ = freq_fraction(H)
    return float((frac > thresh).mean()), frac


def _selfcheck_nd():
    """Settle the multi-index family by PLANTING, then check r=1 reproduces the cyclic path."""
    orders = (4, 10)                       # (Z/33Z)* -- one of n=165's local groups
    e0 = np.arange(orders[0]); e1 = np.arange(orders[1])
    # a pure product character of frequency kappa = (1, 3), as a function of the SUM
    kap = (1, 3)
    ph = lambda a, b: 2 * np.pi * (kap[0] * a / orders[0] + kap[1] * b / orders[1])
    A = ph(e0[:, None], e1[None, :])
    H = np.cos(A[:, :, None, None] + A[None, None, :, :])[..., None]
    frac, bk = freq_fraction_nd(H, orders)
    assert frac[0] > 0.999, frac[0]
    assert bk[0] in (kap, tuple((-k) % o for k, o in zip(kap, orders))), bk[0]
    # a two-character neuron must NOT clear the single-character threshold
    kap2 = (2, 1)
    ph2 = lambda a, b: 2 * np.pi * (kap2[0] * a / orders[0] + kap2[1] * b / orders[1])
    B = ph2(e0[:, None], e1[None, :])
    H2 = (np.cos(A[:, :, None, None] + A[None, None, :, :])
          + np.cos(B[:, :, None, None] + B[None, None, :, :]))[..., None]
    f2, _ = freq_fraction_nd(H2, orders)
    assert 0.45 < f2[0] < 0.55, f2[0]
    # r = 1 must agree with the cyclic path element for element
    s = 12
    a = np.arange(s)
    G = np.cos(2 * np.pi * 3 * (a[:, None] + a[None, :]) / s)[:, :, None]
    fc, kc = freq_fraction(G)
    fn, kn = freq_fraction_nd(G, (s,))
    assert abs(fc[0] - fn[0]) < 1e-12, (fc[0], fn[0])
    assert kn[0] in ((3,), (s - 3,)) and kc[0] == 3, (kc[0], kn[0])
    print("  neurons: product-character family PASS (planted kappa recovered; "
          "r=1 agrees with the cyclic path)")


def _selfcheck():
    _selfcheck_nd()
    s, rng = 60, np.random.default_rng(0)
    a = np.arange(s)
    # a pure single-frequency neuron must read ~1.0 at its own frequency
    for k in (3, 7, 19):
        H = np.cos(2 * np.pi * k * (a[:, None] + a[None, :]) / s)[:, :, None]
        frac, bk = freq_fraction(H)
        assert frac[0] > 0.999 and bk[0] == k, (k, frac[0], bk[0])
    # cos(k a) * cos(k b) -- the product form, all four (+-k, +-k) corners
    H = (np.cos(2 * np.pi * 5 * a / s)[:, None] * np.cos(2 * np.pi * 5 * a / s)[None, :])[:, :, None]
    frac, bk = freq_fraction(H)
    assert frac[0] > 0.999 and bk[0] == 5, (frac[0], bk[0])
    # a two-frequency neuron must NOT clear the single-frequency threshold
    H = (np.cos(2 * np.pi * 4 * (a[:, None] + a[None, :]) / s)
         + np.cos(2 * np.pi * 11 * (a[:, None] - a[None, :]) / s))[:, :, None]
    frac, _ = freq_fraction(H)
    assert 0.45 < frac[0] < 0.55, frac[0]
    # white noise sits near chance, and the shuffle control reproduces that
    H = rng.standard_normal((s, s, 200))
    t, _ = tuned_fraction(H)
    assert t < 0.02, t
    ts, _ = tuned_fraction(shuffle_control(H, seed=1))
    assert ts < 0.02, ts
    # the shuffle must destroy structure it is meant to destroy
    H = np.stack([np.cos(2 * np.pi * 7 * (a[:, None] + a[None, :]) / s)] * 50, -1)
    assert tuned_fraction(H)[0] == 1.0
    assert tuned_fraction(shuffle_control(H, seed=2))[0] < 0.05
    # ... while preserving each neuron's values exactly
    assert np.allclose(np.sort(shuffle_control(H, 3).reshape(-1, 50), 0),
                       np.sort(H.reshape(-1, 50), 0))
    print("neurons: SELFCHECK PASS")


if __name__ == "__main__":
    _selfcheck()
