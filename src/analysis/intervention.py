"""Intervene on a WEIGHT the network reads, then let the network run.

`ablation.py` FFTs the saved logit grid, masks it and inverts. That measures whether the
OUTPUT carries character structure the loss depends on. It cannot show the network
*computes* with those characters, because the mask is applied to the output after the
fact -- and zeroing everything outside the key set in the output partly builds the
"restricted is fine" result in.

This module edits `W_E` in the multiplicative character basis and hands the edited matrix
back to the model. Softmax attention and the ReLU MLP then run on top of it, so nothing
downstream is algebraically forced.

SCOPE, stated rather than silently done: the character basis is defined on UNITS. Rows at
non-units (residue 0, and every non-unit at composite n) and extra token rows (the `=`
token) are PASSED THROUGH UNTOUCHED. A claim from this module is a claim about the unit
part of the embedding.
"""
import numpy as np

from .ablation import _kappa, _neg, _orders
from .transforms import unit_index


def conjugate_closure_1d(freqs, orders):
    """Close a set of single character indices under kappa -> -kappa.

    A real matrix has a conjugate-symmetric spectrum. Masking kappa without -kappa leaves
    an imaginary residue, and `.real` then silently returns something that is not a
    projection of anything. `_selfcheck` asserts both directions: closed sets leave ~0
    imaginary part, unclosed sets leave a large one.
    """
    o = tuple(orders)
    out = set()
    for k in freqs:
        kap = _kappa(k, o)
        out.add(kap)
        out.add(_neg(kap, o))
    return sorted(out)


def character_project(W, n, keep=None, remove=None, keep_dc=True, powers=None):
    """Zero characters of the unit rows of `W`, then return to token coordinates.

    `W` is (n_tokens, d) with row i the embedding of residue i; n_tokens may exceed n.
    Exactly one of `keep` (restricted: everything else zeroed) or `remove` (excluded)
    is given, as an iterable of character indices -- ints when (Z/nZ)* is cyclic,
    tuples otherwise, matching `all_freq_labels`.

    `keep_dc` applies to `keep` only, mirroring `ablation.fourier_ablate`, so the
    restricted arm here and the restricted arm there mean the same thing.
    """
    assert (keep is None) != (remove is None), "give exactly one of keep/remove"
    W = np.asarray(W, dtype=float)
    index, orders, inverse = unit_index(n, powers)
    r = len(orders)

    A = np.zeros(orders + W.shape[1:], dtype=float)
    for x, e in index.items():
        A[e] = W[x]
    F = np.fft.fftn(A, axes=tuple(range(r)))

    spec = conjugate_closure_1d(keep if keep is not None else remove, orders)
    if keep is not None:
        M = np.zeros(orders, dtype=bool)
        for kap in spec:
            M[kap] = True
        if keep_dc:
            M[(0,) * r] = True
    else:
        M = np.ones(orders, dtype=bool)
        for kap in spec:
            M[kap] = False

    A2 = np.fft.ifftn(F * M[(...,) + (None,) * (A.ndim - r)], axes=tuple(range(r)))
    out = np.array(W, dtype=float, copy=True)
    for x, e in index.items():
        out[x] = A2[e].real
    return out


def random_character_sets(key_freqs, n, n_draws, seed=0, powers=None):
    """Draw `n_draws` character sets of the same LABEL COUNT as the key set.

    Deliberately identical to the published null in `analyze_n4.run`, which draws
    `size=len(kf)` from `all_freq_labels(orders)` and closes afterwards. Two consequences
    are inherited on purpose rather than improved on:

    1. The pool INCLUDES the key characters, so a draw may contain one. That makes the
       null more damaging and the resulting p LARGER -- it biases against the finding,
       which is the direction worth inheriting.
    2. Matching is on labels, not on closed components. At n=113 the key set contains the
       self-conjugate Nyquist label 56, so its closure has 7 components where a generic
       4-label draw closes to 8. Excluding key labels or forcing component-parity would
       make this null incomparable to C22/C23, which is a worse defect than the
       one-component asymmetry.
    """
    from .ablation import all_freq_labels
    _, orders, _ = unit_index(n, powers)
    pool = all_freq_labels(orders)
    rng = np.random.default_rng(seed)
    return [[pool[i] for i in rng.choice(len(pool), size=len(key_freqs), replace=False)]
            for _ in range(n_draws)]


def _selfcheck():
    n, d = 113, 8
    index, orders, inverse = unit_index(n)
    r = len(orders)
    rng = np.random.default_rng(0)

    # 1. BIN CONVENTION, SETTLED BY PLANTING -- never by reasoning. Three bin-convention
    #    bugs in this project have been lost to reasoning; argmax is the one readout no
    #    threshold can rescue or break.
    for k0 in (1, 5, 37):
        W = np.zeros((n + 1, 1))
        for x, e in index.items():
            W[x, 0] = np.cos(2 * np.pi * k0 * e[0] / orders[0])
        A = np.zeros(orders + (1,))
        for x, e in index.items():
            A[e] = W[x]
        F = np.abs(np.fft.fftn(A, axes=(0,)))[:, 0]
        got = int(np.argmax(F))
        assert got in (k0, orders[0] - k0), f"planted {k0}, spectrum peaks at {got}"

    # 2. An unmasked round trip is the identity -- the transform itself loses nothing.
    W = rng.standard_normal((n + 1, d))
    every = [k for k in range(orders[0])]
    back = character_project(W, n, keep=every)
    assert np.abs(back - W).max() < 1e-10, np.abs(back - W).max()

    # 3. keep(K, no DC) + remove(K) == W exactly, ON THE UNIT ROWS. The two arms
    #    partition the spectrum, so a bug in either mask shows up here as a residue.
    #    Restricted to units deliberately: non-unit rows pass through in BOTH arms, so
    #    summing them double-counts. The first draft of this check compared all rows and
    #    failed at 1.58 -- the pass-through is correct and the check was wrong.
    K = [2, 8, 47]
    U = np.array(sorted(index))
    a = character_project(W, n, keep=K, keep_dc=False)
    b = character_project(W, n, remove=K)
    assert np.abs(a[U] + b[U] - W[U]).max() < 1e-10, np.abs(a[U] + b[U] - W[U]).max()
    # and the pass-through rows are NOT part of that identity, asserted so a future
    # change that starts transforming them is caught here rather than in a result.
    nonunit = np.array([i for i in range(n) if i not in index])
    assert len(nonunit) and np.abs(a[nonunit] + b[nonunit] - W[nonunit]).max() > 1e-6

    # 4. Non-unit and extra token rows pass through untouched.
    W2 = character_project(W, n, remove=K)
    assert np.array_equal(W2[0], W[0]), "residue 0 was modified"
    assert np.array_equal(W2[n], W[n]), "the '=' token row was modified"

    # 5. A CLOSED set leaves no imaginary residue; an UNCLOSED one leaves a large one.
    #    This is the failure `conjugate_closure_1d` exists to prevent, so assert it fires.
    A = np.zeros(orders + (d,))
    for x, e in index.items():
        A[e] = W[x]
    F = np.fft.fftn(A, axes=(0,))
    for closed, want_small in ((conjugate_closure_1d(K, orders), True),
                               ([_kappa(k, orders) for k in K], False)):
        M = np.ones(orders, dtype=bool)
        for kap in closed:
            M[kap] = False
        resid = np.abs(np.fft.ifftn(F * M[:, None], axes=(0,)).imag).max()
        if want_small:
            assert resid < 1e-12, f"closed set left imaginary residue {resid:.2e}"
        else:
            assert resid > 1e-6, f"UNCLOSED set left only {resid:.2e} -- check 5 is blind"

    # 6. The null draws match the key set's LABEL count, vary, and stay in the pool.
    #    Label-matching, not component-matching, is the published convention -- see
    #    `random_character_sets`. The first draft asserted equal CLOSED sizes and failed,
    #    because 56 is self-conjugate at n=113: its closure has 7 components where a
    #    4-label draw has 8. That asymmetry is real and inherited, not a bug here.
    from .ablation import all_freq_labels
    K4 = [2, 8, 47, 56]
    pool = set(all_freq_labels(orders))
    draws = random_character_sets(K4, n, 50, seed=1)
    for dr in draws:
        assert len(dr) == len(K4), f"draw has {len(dr)} labels, key set has {len(K4)}"
        assert set(dr) <= pool, "a draw left the label pool"
        assert len(set(dr)) == len(dr), "a draw repeated a label"
    assert len({tuple(sorted(dr)) for dr in draws}) > 1, "draws are not varying"
    # The Nyquist asymmetry, asserted so it is a known property and not a surprise later.
    assert len(conjugate_closure_1d([56], orders)) == 1, "56 is not self-conjugate at 113"
    assert len(conjugate_closure_1d([2], orders)) == 2, "2 should close to a pair"

    # 7. keep_dc actually keeps DC. Nothing above exercised it: check 2 keeps every
    #    label INCLUDING 0, and check 3 passes keep_dc=False explicitly, so flipping the
    #    flag to a constant False passed the whole suite. Found by mutation, not review.
    mu = W[U].mean(0)
    with_dc = character_project(W, n, keep=[2, 8], keep_dc=True)
    without = character_project(W, n, keep=[2, 8], keep_dc=False)
    assert np.abs(with_dc[U].mean(0) - mu).max() < 1e-10, "keep_dc=True lost the mean"
    assert np.abs(without[U].mean(0)).max() < 1e-10, "keep_dc=False kept a mean"

    # 8. character_project CLOSES CONJUGATES ITSELF. Check 5 builds its masks by hand, so
    #    deleting the closure inside the function passed the whole suite. Removing label 2
    #    must equal removing {2, 110}; if the function stops closing, the half-masked
    #    spectrum's .real is a different matrix.
    one = character_project(W, n, remove=[2])
    both = character_project(W, n, remove=[2, orders[0] - 2])
    assert np.abs(one - both).max() < 1e-12, "remove=[2] != remove=[2,110]: closure lost"
    #    and it must differ from removing nothing, or check 8 is vacuous.
    assert np.abs(one[U] - W[U]).max() > 1e-6, "removing label 2 changed nothing"

    print("intervention selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
