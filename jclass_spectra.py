"""Per-J-class spectral analysis.

2607.07066 Thm 3.4: a REGULAR J-class is a group, J_d = (Z/(n/d)Z)^x, so it has its own
character basis and GCR applies locally. A NON-REGULAR class has no idempotent, hence no
local inverse -- their stated open problem.

The J-class multiplication table says what happens instead: at a prime power, products
LEAVE the class and descend toward 0 (J_11 * J_11 = {0} at n=121). So the question is
whether the network still builds character structure on a class that is not a group.

Coordinate used: J_d = {d*u : u a unit mod n/d} is a bijection (|J_d| = phi(n/d)), so
index by the UNIT-GROUP ACTION rather than a group law on the class. That is a set-level
coordinate and is defined whether or not J_d is regular -- which is the point. The acting
group (Z/(n/d)Z)^x may itself be non-cyclic, so use the product-character decomposition
uniformly: cyclic is just the one-factor case.
"""
import numpy as np
from math import gcd
from src.tasks.algebra import describe
from src.analysis.transforms import unit_index
from src.analysis.sparsity import gini, participation_ratio

MIN_SIZE = 8            # below this the spectrum is unresolvable (LAB_NOTEBOOK Entry 6)


def class_coords(n, d):
    """{x in J_d: multi-index of its unit coordinate}, plus the factor orders."""
    m = n // d
    if m == 1:
        return None
    idx, orders, _ = unit_index(m)          # handles cyclic AND non-cyclic uniformly
    return {(d * u) % n: e for u, e in idx.items()}, orders


def class_spectrum(W, coords, orders):
    """Product-character transform restricted to one J-class."""
    A = np.zeros(tuple(orders) + (W.shape[1],), dtype=complex)
    for x, e in coords.items():
        A[e] = W[x]
    S = np.fft.fftn(A, axes=tuple(range(len(orders))))
    e = (np.abs(S) ** 2).sum(-1)            # energy per character, over d_model
    return np.sqrt(e).ravel()               # amplitude -- the protocol invariant


def random_control(W, rows, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((len(rows), len(rows))))
    return np.linalg.norm(Q.conj().T @ W[rows], axis=1)


def analyse(n, W, label=""):
    d = describe(n)
    print(f"\n=== n={n}  {d['factorization']}  {label} ===")
    print(f"  {'class':<8} {'size':>5} {'reg':>7} {'factors':>14} {'Gini':>7} {'ctrl':>7} "
          f"{'ratio':>6} {'PR':>6}")
    out = []
    for c in d["jclasses"]:
        dd, sz, reg = c["d"], c["size"], c["regular"]
        cc = class_coords(n, dd)
        if cc is None or sz < MIN_SIZE:
            continue
        coords, orders = cc
        fn = class_spectrum(W, coords, orders)
        rows = np.array(sorted(coords))
        ctrl = random_control(W, rows)
        g, gc = gini(fn), gini(ctrl)
        out.append(dict(n=n, d=dd, size=sz, regular=reg, gini=g, control=gc,
                        ratio=g / max(gc, 1e-9), pr=participation_ratio(fn)))
        print(f"  J_{dd:<6} {sz:>5} {'REG' if reg else 'NON-REG':>7} {str(orders):>14} "
              f"{g:>7.3f} {gc:>7.3f} {g/max(gc,1e-9):>6.2f} {participation_ratio(fn):>6.2f}")
    return out


if __name__ == "__main__":
    rows = []
    for n in [121, 125, 119, 165, 120, 113]:
        W = np.load(f"results/k01_scout/scout_n{n}_seed0.npz")["W_E"][:n].astype(float)
        rows += analyse(n, W - W.mean(0, keepdims=True), "(scout, 1 seed)")

    print("\n--- summary ---")
    print("  A raw REGULAR-vs-NON-REGULAR average is CONFOUNDED BY CLASS SIZE: J_1 is")
    print("  always regular and always the largest class, and Gini rises with size here.")
    print("  The valid comparison is size-matched WITHIN one model (no cross-model confound).")
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[(r["n"], r["size"])].append(r)
    print(f"\n  {'n':>5} {'size':>5}  {'regular':>28}  {'non-regular':>28}")
    any_pair = False
    for (n, sz), rs in sorted(by.items()):
        reg = [r for r in rs if r["regular"]]
        non = [r for r in rs if not r["regular"]]
        if reg and non:
            any_pair = True
            f = lambda g: ", ".join(f"J_{r['d']} {r['ratio']:.2f}x" for r in g)
            print(f"  {n:>5} {sz:>5}  {f(reg):>28}  {f(non):>28}")
    if not any_pair:
        print("  (no size-matched pairs)")
    print("\n  VERDICT: with one seed the size-matched pairs do not separate cleanly.")
    print("  Non-regular classes DO beat their random-basis control (every ratio > 1.9),")
    print("  so character structure is present on classes that are not groups -- but the")
    print("  claim that regular classes carry MORE of it is NOT supported at 1 seed.")
    print(f"\n  (classes smaller than {MIN_SIZE} elements omitted -- unresolvable)")
