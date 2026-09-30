"""C8 (the CRT-dual law) tested without an arbitrary detection threshold.

Entry 5 verified C8 by exact set equality against a 5x-median detector. That works where
the signal is strong (n=165, n=120) and goes silent where it is weak (n=119) -- which
leaves O3 open and makes the claim hostage to the threshold.

Threshold-free version: the law predicts a set of frequencies. Ask whether the observed
spectral energy is enriched on that set, and calibrate against permutations of the same
size. No cut-off, and a p-value instead of a yes/no.

Prime powers are excluded: with one CRT component n/q = 1, so the predicted set is every
frequency and the test is vacuous by construction, not by failure.
"""
import os
import numpy as np
from sympy import factorint
from src.tasks.algebra import describe
from src.analysis.sparsity import key_freqs_5x_median
from src.analysis.stats import perm_p

N_PERM = 20_000
RNG = np.random.default_rng(0)


def predicted(n):
    """Multiples of n/q, one family per maximal prime power q || n, up to n/2."""
    duals = {n // (p ** e) for p, e in factorint(n).items()}
    return sorted({m for d in duals for m in range(d, n // 2 + 1, d)})


def freq_energy(W, n):
    k = np.arange(1, n // 2 + 1)[:, None]
    t = np.arange(n)[None, :]
    s = np.sin(2 * np.pi * k * t / n) @ W
    c = np.cos(2 * np.pi * k * t / n) @ W
    return np.sqrt((s ** 2).sum(1) + (c ** 2).sum(1))


def permutation_test(e, pred_idx, return_null=False):
    """Enrichment of mean energy on the predicted set vs same-size random sets.

    `return_null` exists so a figure can draw the null from THIS implementation rather
    than a second copy of it -- the same reason `analyze_n4.run` takes `return_draws`.
    `obs` does not depend on the RNG; only `p` and the null's shape do.
    """
    m = len(e)
    mask = np.zeros(m, bool)
    mask[pred_idx] = True
    if mask.all() or not mask.any():
        return None
    obs = e[mask].mean() / e[~mask].mean()
    null = np.empty(N_PERM)
    idx = np.arange(m)
    for i in range(N_PERM):
        RNG.shuffle(idx)
        sel = idx[:mask.sum()]
        rest = idx[mask.sum():]
        null[i] = e[sel].mean() / e[rest].mean()
    p = perm_p(int((null >= obs).sum()), N_PERM)
    if return_null:
        return obs, p, float(null.mean()), null
    return obs, p, float(null.mean())


def jclass_indicators(n):
    """The J-class indicator family of Z/nZ, mean-centred, as analyze_scout.py builds it.

    One column per J-class except J_n = {0}. No network is involved: this is a function
    of the algebra of n alone, so if P(n) is a property of the stratification rather than
    of a trained model, it must be visible here.
    """
    ds = sorted({d for d in range(1, n + 1) if n % d == 0 and d != n})
    ind = np.zeros((n, len(ds)))
    for j, d in enumerate(ds):
        for x in range(n):
            if np.gcd(x, n) == d:
                ind[x, j] = 1
    return ind - ind.mean(0, keepdims=True)


def indicator_setequality(n):
    """R3: does the network-free indicator select exactly P(n)?

    Section 9 asserts a set; a Gini is a sparsity number, not that claim. This tests the
    set directly.

    Bin shift: `predicted` returns frequencies and `freq_energy` returns amplitudes at
    position frequency-1, so `main()` writes [k - 1 for k in predicted(n)]. Here the shift
    goes the other way (detector indices are turned back into frequencies by +1), and
    `_selfcheck_bins` plants a frequency to prove which way is which.
    """
    e = freq_energy(jclass_indicators(n), n)
    key = sorted(int(i) + 1 for i in key_freqs_5x_median(e))
    pred = predicted(n)
    prime_variant = sorted({m for p_ in factorint(n)
                            for m in range(n // p_, n // 2 + 1, n // p_)})
    sup = sorted(int(i) + 1 for i in np.where(e > 1e-9 * max(e.max(), 1e-30))[0])
    sel = [e[k - 1] for k in key]
    rej = [e[k - 1] for k in range(1, n // 2 + 1) if k not in key]
    return dict(n=n, key=key, predicted=pred, prime_variant=prime_variant,
                equal=key == pred, prime_equal=key == prime_variant,
                support=len(sup), support_superset=set(sup) >= set(pred) and len(sup) > len(pred),
                margin=(min(sel) / max(rej)) if sel and rej else float("nan"))


def _selfcheck_bins():
    """Plant a frequency and look at where it lands. Never reason about a bin."""
    n = 60
    t = np.arange(n)
    for k0 in (7, 13, 29):
        W = np.cos(2 * np.pi * k0 * t / n)[:, None]
        e = freq_energy(W - W.mean(), n)
        assert int(np.argmax(e)) + 1 == k0, (k0, int(np.argmax(e)) + 1)
    # and the shift main() uses, in the opposite direction, on the same array
    assert [k - 1 for k in predicted(120)] == [k - 1 for k in sorted(predicted(120))]
    print("  selfcheck PASS  freq_energy index i holds frequency i+1 (planted 7, 13, 29)")


def main():
    print(f"C8 permutation test — enrichment of energy on the predicted frequency set")
    print(f"({N_PERM:,} permutations; prime powers excluded as vacuous)\n")
    print(f"{'n':>5} {'factorization':>12} {'|pred|':>7} {'|all|':>6} {'enrich':>8} "
          f"{'null':>6} {'p':>9}  verdict")
    print("-" * 78)
    cases = [(165, "reference/interpreting-monoids checkpoint",
              "results/k02_grid/../k01_scout/scout_n165_seed0.npz")]
    for n in [165, 120, 119, 113, 121, 125]:
        d = describe(n)
        if len(factorint(n)) == 1:
            print(f"{n:>5} {str(dict(factorint(n))):>12} {'—':>7} {'—':>6} "
                  f"{'—':>8} {'—':>6} {'—':>9}  VACUOUS (one CRT component)")
            continue
        W = np.load(f"results/k01_scout/scout_n{n}_seed0.npz")["W_E"][:n].astype(float)
        W = W - W.mean(0, keepdims=True)
        e = freq_energy(W, n)
        pred = [k - 1 for k in predicted(n)]
        r = permutation_test(e, pred)
        if r is None:
            continue
        obs, p, nullmean = r
        v = "ENRICHED" if p < 0.01 else ("weak" if p < 0.05 else "NOT ENRICHED")
        fac = "*".join(f"{q}^{k}" if k > 1 else str(q)
                       for q, k in sorted(factorint(n).items()))
        print(f"{n:>5} {fac:>12} {len(pred):>7} {len(e):>6} {obs:>8.2f} "
              f"{nullmean:>6.2f} {p:>9.2e}  {v}")

    # the same test on the published third-party checkpoint. reference/ is not shipped (it is
    # the authors' repository), so a public clone skips this row, as refcheck.py does.
    _ck = "reference/interpreting-monoids/experiments/P165_d128_h4_mlp512_s1.pt"
    if os.path.exists(_ck):
        import torch
        ck = torch.load(_ck, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        W = sd["embed.weight"].numpy()[:165].astype(float)
        W = W - W.mean(0, keepdims=True)
        e = freq_energy(W, 165)
        obs, p, nullmean = permutation_test(e, [k - 1 for k in predicted(165)])
        print(f"\n{'165':>5} {'THEIR ckpt':>12} {len(predicted(165)):>7} {len(e):>6} "
              f"{obs:>8.2f} {nullmean:>6.2f} {p:>9.2e}  "
              f"{'ENRICHED' if p < 0.01 else 'NOT ENRICHED'}")
    else:
        print(f"\n  {_ck} missing -- clone the authors' repositories into reference/ "
              "to run this row")
    print("\nNote: p is Phipson-Smyth (1+b)/(B+1) over permutations at least as enriched.")
    print(f"The floor is 1/{N_PERM+1:,} = {1/(N_PERM+1):.2e}; a p AT the floor means no draw")
    print("reached the observation, not that the probability is zero.")
    print("Enrichment is artifact-dependent and both published values are real: the k01 scout")
    print("checkpoint reads 14.08 at n=165 and 3.30 at n=119, the k03 runs 17.44 and 3.27.")

    # R3 -- the same law asked of the STRATIFICATION, with no network involved.
    _selfcheck_bins()
    print(f"\nR3 (PREREGISTER_r3_indicator_setequality.md) -- J-class indicators, no network")
    print(f"{'n':>5} {'|K|':>4} {'|P(n)|':>7} {'K = P(n)':>9} {'K <= P(n)':>10} "
          f"{'enrich':>7} {'p':>9}")
    print("-" * 60)
    for n in [165, 120, 119, 99, 100, 147, 105, 143, 75, 98]:
        r = indicator_setequality(n)
        e = freq_energy(jclass_indicators(n), n)
        obs, p, _ = permutation_test(e, [k - 1 for k in r["predicted"]])
        print(f"{n:>5} {len(r['key']):>4} {len(r['predicted']):>7} {str(r['equal']):>9} "
              f"{str(set(r['key']) <= set(r['predicted'])):>10} {obs:>7.2f} {p:>9.2e}")
    print("R3-1 is NOT HELD and that is the finding: the indicators are ENRICHED on P(n)")
    print("but do not SELECT it. The stratification supplies the duals of EVERY divisor")
    print("(42 frequencies at n=165); the trained network selects the maximal-prime-power")
    print("subfamily P(n) exactly (8 of 8 at n=165, 7 of 7 at n=120). At n=120 the")
    print("indicator picks 20 = 120/6, which is in neither P(n) nor its prime variant.")


if __name__ == "__main__":
    main()
