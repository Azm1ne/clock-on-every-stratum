"""Analysis bases for functions on Z/nZ.

The methodological claim (2606.17399) is that sparsity is basis-dependent: modular
multiplication looks dense in the additive DFT and sparse in the multiplicative
character basis. So we need every basis, plus a random one as a control.

(Z/nZ)* is a finite abelian group, so it decomposes as a product of cyclic factors.
Handling that product uniformly covers both the cyclic case (113, 121, 125 -- one
factor) and the non-cyclic case (119, 120), with one code path and one n-dim FFT.
"""
import numpy as np
from math import gcd
from sympy import factorint

from src.tasks.algebra import units, primitive_root


def _component_generators(n):
    """Generators of (Z/nZ)* with their orders, via CRT over prime-power components.

    Returns [(g, order), ...] with every unit uniquely expressible as prod g_i^e_i.
    """
    gens = []
    for p, a in sorted(factorint(n).items()):
        q = p ** a
        m = n // q
        # lift a generator mod q to mod n: = gen mod q, = 1 mod n/q
        lift = (lambda h: h if m == 1 else
                (h * m * pow(m, -1, q) + 1 * q * pow(q, -1, m)) % n)
        if p == 2:
            if a == 1:
                continue                       # trivial group
            elif a == 2:
                gens.append((lift(3), 2))
            else:
                gens.append((lift(q - 1), 2))          # -1
                gens.append((lift(5), 2 ** (a - 2)))   # 5 generates the rest
        else:
            g = primitive_root(q)
            gens.append((lift(g), q - q // p))         # phi(p^a)
    return gens


def unit_index(n, powers=None):
    """Bijection unit x <-> exponent tuple (e_1,...,e_k) w.r.t. the component generators.

    This is the discrete logarithm, generalized to non-cyclic unit groups.
    Returns (index: {x: tuple}, orders: tuple, inverse: {tuple: x}).

    `powers` replaces each component generator g_i by g_i**t_i -- the SAME subgroup, a
    different labelling of it. It exists so the choice of generator can be varied and the
    downstream readout checked for equivariance: nothing we claim about a circuit may
    depend on which primitive root we happened to pick. Each t_i must be coprime to its
    component order or the map is not a bijection; the assertions below catch it.
    """
    gens = _component_generators(n)
    if powers is not None:
        assert len(powers) == len(gens), f"n={n}: need {len(gens)} powers, got {len(powers)}"
        gens = [(pow(g, int(t), n), o) for (g, o), t in zip(gens, powers)]
    orders = tuple(o for _, o in gens)
    index, inverse = {}, {}
    for e in np.ndindex(*orders) if orders else [()]:
        x = 1
        for (g, _), ei in zip(gens, e):
            x = x * pow(g, int(ei), n) % n
        index[x] = tuple(int(i) for i in e)
        inverse[tuple(int(i) for i in e)] = x
    U = units(n)
    assert len(index) == len(U), f"n={n}: index covers {len(index)} of {len(U)} units"
    assert set(index) == set(U), f"n={n}: index is not the unit group"
    return index, orders, inverse


def additive_dft(W):
    """DFT along the token axis. W is (n,) or (n, d)."""
    return np.fft.fft(np.asarray(W, dtype=complex), axis=0)


def character_transform(W, n, powers=None):
    """Multiplicative character transform: reindex units by discrete log, then FFT.

    For cyclic (Z/nZ)* this is the multiplicative character basis of 2606.17399;
    for non-cyclic it is the product-character basis. Only defined on units, so
    rows of W at non-units are dropped -- that loss is itself the object of study.

    `powers` picks a different generator for each cyclic component (see `unit_index`).
    """
    W = np.asarray(W, dtype=complex)
    index, orders, _ = unit_index(n, powers)
    tail = W.shape[1:]
    A = np.zeros(orders + tail, dtype=complex)
    for x, e in index.items():
        A[e] = W[x]
    return np.fft.fftn(A, axes=tuple(range(len(orders))))


def random_orthogonal(W, seed=0):
    """Control basis: 'sparse in basis X' means nothing without this comparison."""
    W = np.asarray(W, dtype=complex)
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((W.shape[0], W.shape[0])))
    return Q.conj().T @ W


def energy(spectrum, fold_conjugate=True):
    """Energy per frequency, summed over the feature axis.

    Real inputs give conjugate-symmetric spectra, so frequency f and -f carry the
    same energy. Folding them avoids double-counting the number of 'key frequencies'.
    """
    S = np.asarray(spectrum)
    k = S.ndim - 1 if S.ndim > 1 else S.ndim
    e = np.abs(S) ** 2
    if e.ndim > (1 if k else 0):
        e = e.reshape(e.shape[:k] + (-1,)).sum(axis=-1) if k else e
    if not fold_conjugate:
        return e.ravel()
    seen, out = set(), []
    for idx in np.ndindex(*e.shape):
        conj = tuple((-i) % s for i, s in zip(idx, e.shape))
        if conj in seen:
            continue
        seen.add(idx)
        out.append(e[idx] + (e[conj] if conj != idx else 0.0))
    return np.array(out)
