"""Nothing we claim about a circuit may depend on which primitive root we happened to pick.

Every multiplicative-basis result in this project -- C4, C6, C21, C22, C23 -- is read in
exponent coordinates, and those coordinates need a choice of generator. `primitive_root(n)`
returns the smallest one, and every analysis uses it. If the readout moved with
that arbitrary choice, the readout would be an artefact.

It does not, and the invariance is exact rather than approximate. Replacing g by g^t
(gcd(t, phi) = 1) is a relabelling of the same group, so:

  * the exponent map obeys  dlog_{g^t}(x) = t^{-1} . dlog_g(x)  (mod phi);
  * the spectrum is permuted, so Gini and the participation ratio are invariant;
  * a key frequency moves exactly by  k -> t.k (mod phi); it does not stay put;
  * the (k,k) restricted diagonal maps to (tk,tk), still the diagonal, so baseline,
    restricted and excluded losses are invariant.

Non-cyclic unit groups have no single generator, but the same holds per cyclic component.

This test also showed that the losses are invariant to machine precision while the ratio
excluded / mean(random control) is not a stable quantity, which is why the permutation p
replaced it (see the header of `analyze_n4.py`).
"""
import sys, os
import numpy as np
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sympy import n_order, totient
from src.analysis.transforms import unit_index, character_transform, energy
from src.analysis.sparsity import gini, participation_ratio
import analyze_n4 as A

D = "results/k03_grid_acts"
CYCLIC = [(113, "B_thesis_n113_s0"), (121, "B_thesis_n121_s0"), (125, "B_thesis_n125_s0")]
NONCYCLIC = [(119, "B_thesis_n119_s0"), (120, "B_thesis_n120_s0"), (165, "B_thesis_n165_s0")]
TOL = 1e-10


def coprime_powers(m, k=4):
    return [t for t in range(3, m) if gcd(t, m) == 1][:k]


def fold(k, m):
    return min(k % m, (-k) % m)


def check_index_law(n):
    """dlog_{g^t}(x) = t^{-1} dlog_g(x) mod phi -- the algebra, before any model."""
    i0, (phi,), _ = unit_index(n)
    for t in coprime_powers(phi, 3):
        it, _, _ = unit_index(n, [t])
        tinv = pow(t, -1, phi)
        bad = [x for x in i0 if it[x][0] != (tinv * i0[x][0]) % phi]
        assert not bad, f"n={n} t={t}: index law fails at {bad[:5]}"
    print(f"  n={n:<4} exponent map obeys dlog_(g^t) = t^-1 . dlog_g       OK")


def spectra(n, tag, powers=None):
    W = np.load(f"{D}/WE_{tag}.npz", allow_pickle=False)["W_E"][:n].astype(float)
    amp = np.sqrt(energy(character_transform(W - W.mean(0, keepdims=True), n, powers)))
    return gini(amp), participation_ratio(amp ** 2)


def main():
    print("Generator equivariance -- is the multiplicative readout an artefact of g?\n")
    print("A. the algebra, independent of any model")
    for n, _ in CYCLIC:
        check_index_law(n)

    print("\nB. cyclic moduli: spectrum statistics and the ablation, k03 seed 0")
    for n, tag in CYCLIC:
        phi = int(totient(n))
        g0, p0 = spectra(n, tag)
        base = A.run(tag, n, verbose=False)
        assert base is not None, tag
        for t in coprime_powers(phi, 4):
            gt, pt = spectra(n, tag, [t])
            assert abs(gt - g0) < TOL, f"n={n} t={t}: Gini_mult moved {g0}->{gt}"
            assert abs(pt - p0) < TOL * max(1, p0), f"n={n} t={t}: PR moved {p0}->{pt}"
            r = A.run(tag, n, powers=[t], verbose=False)
            for f in ("baseline", "restricted", "excluded"):
                assert abs(r[f] - base[f]) <= TOL * max(abs(base[f]), 1e-12), \
                    f"n={n} t={t}: {f} moved {base[f]:.6e} -> {r[f]:.6e}"
            want = sorted({fold(t * k, phi) for k in base["key"]})
            assert sorted(r["key"]) == want, \
                f"n={n} t={t}: key set {sorted(r['key'])} != predicted k->t.k {want}"
            assert r["p_perm"] == base["p_perm"] or (r["p_perm"] < 0.01) == (base["p_perm"] < 0.01), \
                f"n={n} t={t}: permutation verdict flipped"
        print(f"  n={n:<4} phi={phi:<4} Gini {g0:.4f} PR {p0:.2f} invariant over "
              f"g^{coprime_powers(phi, 4)}; keys map by k->t.k; losses to {TOL:.0e}   OK")

    print("\nC. non-cyclic moduli: the same, per cyclic component")
    for n, tag in NONCYCLIC:
        _, orders, _ = unit_index(n)
        base = A.run(tag, n, verbose=False)
        assert base is not None, tag
        # twist one component at a time, then all of them together
        twists = [[1] * len(orders) for _ in orders]
        for i, o in enumerate(orders):
            c = coprime_powers(o, 1)
            twists[i][i] = c[0] if c else 1
        twists.append([coprime_powers(o, 1)[0] if coprime_powers(o, 1) else 1
                       for o in orders])
        for pw in twists:
            if all(t == 1 for t in pw):
                continue
            r = A.run(tag, n, powers=pw, verbose=False)
            for f in ("baseline", "restricted", "excluded"):
                assert abs(r[f] - base[f]) <= TOL * max(abs(base[f]), 1e-12), \
                    f"n={n} powers={pw}: {f} moved {base[f]:.6e} -> {r[f]:.6e}"
            assert r["n_key"] == base["n_key"], \
                f"n={n} powers={pw}: key COUNT moved {base['n_key']} -> {r['n_key']}"
            assert (r["p_perm"] < 0.01) == (base["p_perm"] < 0.01), \
                f"n={n} powers={pw}: permutation verdict flipped"
        grp = "x".join(f"Z_{o}" for o in orders)
        print(f"  n={n:<4} {grp:<16} losses and key count invariant over "
              f"{len(twists)-1} component twists   OK")

    print("\nD. the statistic C22/C23 reports must itself be stable")
    n, tag = 121, "B_thesis_n121_s0"
    phi = int(totient(n))
    base = A.run(tag, n, verbose=False)
    sm = [base["sep_median"]] + [A.run(tag, n, powers=[t], verbose=False)["sep_median"]
                                 for t in coprime_powers(phi, 4)]
    sm = np.array(sm)
    spread = (sm.max() - sm.min()) / sm.mean()
    print(f"  excluded / MEDIAN control over 5 generators: spread {spread:.1%}")
    assert spread < 0.05, f"median-control separation is not stable: {sm}"
    print("  median-control separation stable under every generator OK")

    print("\ntest_generator_equivariance: PASS")


if __name__ == "__main__":
    main()
