"""TDD for the ablation harness: build a logit grid whose mechanism we KNOW, then assert
the harness attributes it to the right frequencies and not to the others.

If this passes, an ablation result on a real model means something. If it does not, every
downstream causal claim is decoration.
"""
import numpy as np
from src.analysis.ablation import (fourier_ablate, key_freq_pairs, ablation_report,
                                   cross_entropy_np)

N, NF = 41, 3
rng = np.random.default_rng(0)
KEY = sorted(rng.choice(np.arange(1, N // 2), size=NF, replace=False).tolist())


def synth_grid():
    """A 'model' that computes (a+b) mod N using exactly the frequencies in KEY --
    a Clock circuit, by construction."""
    a = np.arange(N)[:, None, None]
    b = np.arange(N)[None, :, None]
    c = np.arange(N)[None, None, :]
    g = np.zeros((N, N, N))
    for k in KEY:
        g += np.cos(2 * np.pi * k * (a + b - c) / N)      # peaks where c == a+b mod N
    return g * 6.0


def main():
    print(f"synthetic Clock circuit on n={N}, key frequencies {KEY}")
    g = synth_grid()
    y = ((np.arange(N)[:, None] + np.arange(N)[None, :]) % N).ravel()
    flat = lambda x: x.reshape(N * N, -1)

    acc = (flat(g).argmax(-1) == y).mean()
    print(f"  synthetic grid accuracy {acc:.4f}   {'OK' if acc == 1.0 else 'FAIL'}")
    assert acc == 1.0, "synthetic circuit is wrong; test is invalid"

    rep = ablation_report(g, y, KEY, N)
    print(f"  baseline loss   {rep['baseline']:.4e}")
    print(f"  restricted loss {rep['restricted']:.4e}  (keep only key freqs)")
    print(f"  excluded loss   {rep['excluded']:.4e}  (remove only key freqs)")

    # restricted must not hurt: the key frequencies ARE the whole mechanism here
    assert rep["restricted"] <= rep["baseline"] * 1.01 + 1e-9, rep
    print("  restricted <= baseline                       OK")

    # excluded must be catastrophic: removing the mechanism must remove the performance
    assert rep["excluded"] > rep["baseline"] + 1.0, rep
    print(f"  excluded >> baseline (+{rep['excluded']-rep['baseline']:.2f})            OK")

    # per-frequency: key frequencies hurt when ablated, non-key ones do not
    dk = [rep["per_freq"][k] for k in KEY]
    dn = [v for k, v in rep["per_freq"].items() if k not in KEY]
    print(f"  mean delta, key freqs      {np.mean(dk):+.4f}")
    print(f"  max  delta, non-key freqs  {np.max(dn):+.2e}")
    assert min(dk) > 0.1, f"a key frequency did not matter: {dict(zip(KEY, dk))}"
    assert max(dn) < 1e-6, f"a non-key frequency mattered: {max(dn)}"
    print("  key freqs matter, non-key freqs do not       OK")

    # conjugate closure: ablating without it leaves an imaginary residue
    pairs = key_freq_pairs([KEY[0]], N)
    neg = lambda t: tuple((-x) % N for x in t)
    assert all((neg(ka), neg(kb)) in set(pairs) for ka, kb in pairs)
    print("  conjugate closure is closed                  OK")

    # a real grid must come back real after a no-op ablation
    err = np.abs(fourier_ablate(g, remove=[]) - g).max()
    assert err < 1e-9, err
    print(f"  no-op ablation is identity (err {err:.1e})     OK")

    # A precomputed spectrum must be bit-identical, not merely close. The permutation
    # null re-ablates one constant grid B times, so the forward FFT is hoisted out of the
    # loop; if that changed any bit, every published p would come from different code.
    # Exact equality, not a tolerance.
    from src.analysis.ablation import fourier_spectrum
    F = fourier_spectrum(g, 1)
    pr = key_freq_pairs(KEY[:2], N)
    for kw in ({"remove": pr}, {"keep": pr}):
        a = fourier_ablate(g, **kw)
        b = fourier_ablate(g, spectrum=F, **kw)
        assert np.array_equal(a, b), np.abs(a - b).max()
    print("  precomputed spectrum is bit-identical         OK")

    print("  --- product characters (N4b) ---")
    main_nd()

    print("\ntest_ablation: PASS")


# ---------------------------------------------------------------- N4b
# The same test where the index group is NOT cyclic. (Z/35)* = Z_4 x Z_6, so a character
# is a multi-index and the logit grid is a 4-dimensional array over (e_a, e_b). If this
# passes, an ablation at n=119 (Z_6 x Z_16) or n=120 (Z_2^3 x Z_4) means something.

ND_N = 35


def _neg_lbl(k, orders):
    return tuple((-x) % o for x, o in zip(k, orders))


def synth_grid_nd(orders, key):
    """A 'model' computing a*b mod n on the units, using exactly the product characters
    in `key`. Units are indexed by exponent tuple, and multiplication of units IS
    addition of exponent tuples -- so this is the clock, in the only coordinates where
    n has one."""
    E = np.array(list(np.ndindex(*orders)))                  # (phi, r) exponent tuples
    w = 2 * np.pi / np.array(orders, float)
    g = np.zeros((len(E), len(E), len(E)))
    for k in key:
        ph = (E * (np.array(k) * w)).sum(1)                  # phase of each exponent tuple
        g += np.cos(ph[:, None, None] + ph[None, :, None] - ph[None, None, :])
    return (g * 6.0).reshape(orders + orders + (len(E),))


def main_nd():
    from src.analysis.transforms import unit_index
    from src.analysis.ablation import all_freq_labels
    index, orders, inverse = unit_index(ND_N)
    E = [tuple(e) for e in np.ndindex(*orders)]
    labels = all_freq_labels(orders)
    # The key set must GENERATE the dual group, or no clock built from it can be exact:
    # sum_k cos(chi_k(x)) is maximal only at x = identity when the k's have trivial
    # common kernel. [(0,2),(1,1)] does not generate Z_4 x Z_6 and gives accuracy 0.500.
    # That is a real constraint on reading a model, not an artefact of this test.
    key = [(1, 0), (0, 1), (1, 1)]
    assert all(k in labels or _neg_lbl(k, orders) in labels for k in key)
    print(f"  n={ND_N}, (Z/nZ)* = {' x '.join('Z_%d' % o for o in orders)}, "
          f"key product characters {key}")

    g = synth_grid_nd(orders, key)
    pos = {e: i for i, e in enumerate(E)}
    y = np.array([pos[tuple((np.array(a) + np.array(b)) % np.array(orders))]
                  for a in E for b in E])
    npts = len(E)
    acc = (g.reshape(npts * npts, -1).argmax(-1) == y).mean()
    print(f"  synthetic product-character grid accuracy {acc:.4f}   "
          f"{'OK' if acc == 1.0 else 'FAIL'}")
    assert acc == 1.0, "synthetic product-character circuit is wrong; test is invalid"

    # the exponent grid really is a*b mod n, not just an abstract group law
    a0, b0 = E[3], E[5]
    assert inverse[a0] * inverse[b0] % ND_N == inverse[
        tuple((np.array(a0) + np.array(b0)) % np.array(orders))]
    print("  exponent addition == multiplication mod n     OK")

    rep = ablation_report(g, y, key, orders)
    print(f"  baseline {rep['baseline']:.4e}  restricted {rep['restricted']:.4e}  "
          f"excluded {rep['excluded']:.4e}")
    assert rep["restricted"] <= rep["baseline"] * 1.01 + 1e-9, rep
    assert rep["excluded"] > rep["baseline"] + 1.0, rep
    dk = [rep["per_freq"][k] for k in key]
    dn = [v for k, v in rep["per_freq"].items() if k not in key]
    print(f"  mean delta key {np.mean(dk):+.4f}   max delta non-key {np.max(dn):+.2e}")
    # The absolute delta scales with group order and logit scale (0.38 at n=41 with 3
    # freqs over Z_41, 0.05-0.10 here over Z_4 x Z_6). What discriminates is the gap to
    # the non-key ceiling, which is at machine epsilon.
    assert min(dk) > 0.01 and min(dk) > 1e6 * max(max(dn), 1e-30), \
        f"a key product character did not matter: {dict(zip(key, dk))}"
    assert max(dn) < 1e-6, f"a non-key product character mattered: {max(dn)}"
    print("  key characters matter, non-key do not         OK")

    # The invariant that makes the numbers comparable: the cyclic path is the r=1 case
    # of the same code. Same grid, same answer, whether n is passed as a scalar or as a
    # one-element tuple of orders with tuple frequency labels.
    g1 = synth_grid()
    y1 = ((np.arange(N)[:, None] + np.arange(N)[None, :]) % N).ravel()
    r_scalar = ablation_report(g1, y1, KEY, N)
    r_tuple = ablation_report(g1.reshape((N, N, N)), y1, [(k,) for k in KEY], (N,))
    for f in ("baseline", "restricted", "excluded"):
        assert abs(r_scalar[f] - r_tuple[f]) < 1e-12, (f, r_scalar[f], r_tuple[f])
    print("  scalar n and orders=(n,) agree exactly        OK")


if __name__ == "__main__":
    main()
