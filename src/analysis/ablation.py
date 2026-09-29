"""Fourier-space ablation of the logits (Nanda 2301.05217 section 4.4).

Sparsity says a basis *describes* the model; ablation says a component is *responsible*
for it. Every "X is the mechanism" claim in this project has to route through here.

Method: logits over all (a,b) pairs form an (n, n, n_out) tensor. Take a 2D DFT over the
two input axes, zero selected frequency components, invert, and measure the loss change.

PRODUCT CHARACTERS (N4b). (Z/nZ)* is abelian but not always cyclic: at n=119 it is
Z_6 x Z_16, at n=120 it is Z_2^3 x Z_4. There is then no discrete log and no single
frequency index -- a character is a MULTI-index kappa = (k_1..k_r), and the grid over
(unit_a, unit_b) is a 2r-dimensional array, not a matrix. Every function here therefore
takes `n` either as a scalar (cyclic: orders = (n,)) or as the tuple of component orders,
and a frequency label is an int in the first case and a tuple in the second. The cyclic
path is the r=1 special case of the same code, not a separate one -- which is what makes
the numbers comparable across moduli.
"""
import numpy as np


def _orders(n):
    """Component orders of the index group: scalar n means the cyclic group Z_n."""
    return (int(n),) if np.isscalar(n) else tuple(int(x) for x in n)


def _kappa(k, orders):
    """A frequency label (int, or tuple for a product character) as a reduced tuple."""
    t = (k,) if np.isscalar(k) else tuple(k)
    assert len(t) == len(orders), f"frequency {k} does not match orders {orders}"
    return tuple(int(x) % o for x, o in zip(t, orders))


def _neg(kap, orders):
    return tuple((-x) % o for x, o in zip(kap, orders))


def logits_grid(logits, n):
    """(npts*npts, n_out) in row-major (a,b) order -> orders + orders + (n_out,)."""
    o = _orders(n)
    return np.asarray(logits).reshape(o + o + (-1,))


def fourier_spectrum(grid, r):
    """The forward transform `fourier_ablate` needs, so a permutation loop computes it once.

    A permutation null re-ablates ONE constant grid B times, and the forward FFT does not
    depend on the draw: at B = 25,000 the hoisted version is the difference between one
    transform and 25,000 identical ones. The arithmetic is unchanged -- same input, same
    deterministic call -- so results are bit-identical, which `_selfcheck` asserts.
    """
    return np.fft.fftn(grid, axes=tuple(range(2 * r)))


def fourier_ablate(grid, keep=None, remove=None, keep_dc=True, spectrum=None):
    """Zero frequency components of a logit grid, then invert.

    `keep`/`remove` are iterables of (kappa_a, kappa_b) index pairs -- as produced by
    `conjugate_closure` / `key_freq_pairs`, which always emit tuples. Exactly one must be
    given: `keep` = restricted (everything else zeroed), `remove` = excluded.
    `spectrum` is an optional precomputed `fourier_spectrum(grid, r)` for the same grid.
    """
    assert (keep is None) != (remove is None), "give exactly one of keep/remove"
    spec = list(keep if keep is not None else remove)
    r = len(spec[0][0]) if spec else (grid.ndim - 1) // 2      # index-group rank
    axes = tuple(range(2 * r))
    F = fourier_spectrum(grid, r) if spectrum is None else spectrum
    shape = grid.shape[:2 * r]
    if keep is not None:
        M = np.zeros(shape, dtype=bool)
        for ka, kb in spec:
            M[tuple(ka) + tuple(kb)] = True
        if keep_dc:
            M[(0,) * (2 * r)] = True
    else:
        M = np.ones(shape, dtype=bool)
        for ka, kb in spec:
            M[tuple(ka) + tuple(kb)] = False
    return np.fft.ifftn(F * M[..., None], axes=axes).real


def conjugate_closure(pairs, n):
    """A real signal's spectrum is conjugate-symmetric: (ka,kb) and (-ka,-kb) carry the
    same information. Ablating one without the other leaves an imaginary residue and a
    meaningless loss. Always close the set before ablating."""
    o = _orders(n)
    out = set()
    for ka, kb in pairs:
        a, b = _kappa(ka, o), _kappa(kb, o)
        out.add((a, b))
        out.add((_neg(a, o), _neg(b, o)))
    return sorted(out)


def key_freq_pairs(freqs, n):
    """Nanda's restricted set: for each key frequency k, the components carrying
    cos/sin(w_k(a+b)) live at (k,k) and its conjugate (-k,-k). For a product character
    the same diagonal holds multi-index-wise, because multiplication of units is still
    addition of exponent tuples."""
    return conjugate_closure([(k, k) for k in freqs], n)


def all_freq_labels(n, include_dc=False):
    """Every frequency label worth ablating: conjugate pairs folded so each is tested
    once, DC dropped unless asked for. Ints when cyclic, tuples for product characters.

    With include_dc=True the order matches `transforms.energy(..., fold_conjugate=True)`
    element for element, so an energy vector can be labelled by zipping the two.
    """
    o = _orders(n)
    if len(o) == 1:
        return list(range(0 if include_dc else 1, o[0] // 2 + 1))
    seen, out = set(), []
    for idx in np.ndindex(*o):
        if (idx == (0,) * len(o) and not include_dc) or idx in seen:
            continue
        seen.add(idx); seen.add(_neg(idx, o))
        out.append(idx)
    return out


def cross_entropy_np(logits, targets):
    z = logits - logits.max(-1, keepdims=True)
    return float(-(z[np.arange(len(targets)), targets]
                   - np.log(np.exp(z).sum(-1))).mean())


def ablation_report(grid, targets, key_freqs, n):
    """Restricted / excluded / baseline loss, plus per-frequency ablation deltas."""
    o = _orders(n)
    npts = int(np.prod(o))
    flat = lambda g: g.reshape(npts * npts, -1)
    base = cross_entropy_np(flat(grid), targets)
    keep = key_freq_pairs(key_freqs, n)
    restricted = cross_entropy_np(flat(fourier_ablate(grid, keep=keep)), targets)
    excluded = cross_entropy_np(flat(fourier_ablate(grid, remove=keep)), targets)
    per_freq = {}
    for k in all_freq_labels(n):
        pk = key_freq_pairs([k], n)
        per_freq[k] = cross_entropy_np(flat(fourier_ablate(grid, remove=pk)), targets) - base
    return dict(baseline=base, restricted=restricted, excluded=excluded, per_freq=per_freq)


def unit_logit_grid(z, n, powers=None):
    """A saved checkpoint's logits as an exponent-ordered grid, with its key characters.

    Extracted from `analyze_n4.run` so a figure can draw the very spectrum the causal
    test ablates instead of a second implementation of it. Two implementations of this
    would be two answers to "which characters are the key set", and one of them would
    be wrong.

    Returns (G, y, orders, key_freqs) where G has shape orders + orders + (n_out,), `y`
    is the true label per flattened (a, b) cell, and `key_freqs` are the >5x-median
    characters of the embedding, DC excluded.
    """
    from math import gcd
    import numpy as _np
    from src.analysis.transforms import unit_index, character_transform, energy
    from src.analysis.sparsity import key_freqs_5x_median

    index, orders, inverse = unit_index(n, powers)
    E = list(_np.ndindex(*orders))
    L = z["logits_all"]
    side = int(round(_np.sqrt(L.shape[0])))
    grid_vals = (_np.arange(n) if side == n
                 else _np.array([x for x in range(n) if gcd(x, n) == 1]))
    pos = {int(v): i for i, v in enumerate(grid_vals)}
    sel = _np.array([pos[inverse[e]] for e in E])
    G = L.reshape(side, side, -1)[_np.ix_(sel, sel)].astype(float)
    G = G.reshape(orders + orders + (-1,))
    u = _np.array([inverse[e] for e in E])
    y = ((u[:, None] * u[None, :]) % n).ravel()

    W = z["W_E"][:n].astype(float)
    amp = _np.sqrt(energy(character_transform(W - W.mean(0, keepdims=True), n, powers)))
    labels = all_freq_labels(orders, include_dc=True)
    assert len(labels) == len(amp), (len(labels), len(amp))
    kf = [labels[i] for i in key_freqs_5x_median(amp) if i != 0]
    return G, y, orders, kf
