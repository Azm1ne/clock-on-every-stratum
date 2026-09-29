"""Validate the analysis basis machinery against signals whose answer we already know.

If the character transform is wrong, every downstream sparsity claim is wrong, and we
would not find out until we misread a real embedding in week 8. So: synthesize an
embedding that IS a sparse multiplicative-character signal, and assert the instrument
reports exactly that -- and that the additive DFT, correctly, sees noise.
"""
import numpy as np
from src.tasks.algebra import units, PRIMARY, REPLICATION
from src.analysis.transforms import (unit_index, additive_dft, character_transform,
                                     random_orthogonal, energy)
from src.analysis.sparsity import gini, participation_ratio, n_key_frequencies

D, K, SEED = 128, 4, 0     # d_model, number of planted frequencies


def synth_multiplicative(n, k=K, seed=SEED):
    """Embedding whose rows are k multiplicative characters evaluated at each unit."""
    rng = np.random.default_rng(seed)
    index, orders, _ = unit_index(n)
    freqs = [tuple(rng.integers(1, o) for o in orders) for _ in range(k)]
    coef = rng.standard_normal((k, D))
    W = np.zeros((n, D))
    for x, e in index.items():
        for j, f in enumerate(freqs):
            phase = 2 * np.pi * sum(fi * ei / oi for fi, ei, oi in zip(f, e, orders))
            W[x] += np.cos(phase) * coef[j]
    return W


def synth_additive(n, k=K, seed=SEED):
    """Embedding that is k standard (additive) Fourier frequencies -- the mirror case."""
    rng = np.random.default_rng(seed)
    freqs = rng.choice(np.arange(1, n // 2), size=k, replace=False)
    coef = rng.standard_normal((k, D))
    x = np.arange(n)[:, None]
    return sum(np.cos(2 * np.pi * f * x / n) * coef[j] for j, f in enumerate(freqs))


def report(name, n, W):
    gm = gini(energy(character_transform(W, n)))
    ga = gini(energy(additive_dft(W)))
    gr = gini(energy(random_orthogonal(W, seed=SEED)))
    km = n_key_frequencies(energy(character_transform(W, n)))
    ka = n_key_frequencies(energy(additive_dft(W)))
    print(f"  {name:<26} n={n:<4} Gini mult={gm:.3f}  add={ga:.3f}  rand={gr:.3f}"
          f"   keyfreqs mult={km:<3} add={ka}")
    return gm, ga, gr, km, ka


def main():
    print("unit_index is a bijection onto the unit group:")
    for n in PRIMARY + REPLICATION:
        index, orders, inv = unit_index(n)
        assert len(index) == len(units(n)) == int(np.prod(orders))
        assert all(inv[e] == x for x, e in index.items())
        print(f"  n={n:<4} factors={orders}  |U|={len(index)}")

    print("\nplanted MULTIPLICATIVE signal -> sparse in mult basis, dense elsewhere:")
    for n in PRIMARY:
        gm, ga, gr, km, ka = report("mult signal", n, synth_multiplicative(n))
        assert gm > ga and gm > gr, f"n={n}: mult basis failed to win ({gm} vs {ga}, {gr})"
        assert km <= 2 * K, f"n={n}: expected <={2*K} key freqs (conj. pairs), got {km}"
        assert ka > km, f"n={n}: additive basis should be denser, got {ka} vs {km}"

    print("\nplanted ADDITIVE signal -> sparse in additive basis (the mirror control):")
    for n in PRIMARY:
        gm, ga, gr, km, ka = report("add signal", n, synth_additive(n))
        assert ga > gm, f"n={n}: additive basis failed to win ({ga} vs {gm})"
        assert ka <= 2 * K, f"n={n}: expected <={2*K} key freqs, got {ka}"

    print("\nwhite noise -> no basis wins (guards against a metric that always says 'sparse'):")
    rng = np.random.default_rng(SEED)
    for n in [113, 121]:
        W = rng.standard_normal((n, D))
        gm, ga, gr, _, _ = report("noise", n, W)
        assert max(gm, ga, gr) < 0.35, f"n={n}: spurious sparsity {gm},{ga},{gr}"

    print("\nparticipation ratio tracks the plant count:")
    for n in [113, 121, 125]:
        pr = participation_ratio(energy(character_transform(synth_multiplicative(n), n)))
        assert pr < 2 * K + 1, f"n={n}: PR={pr:.2f} too high for {K} planted freqs"
        print(f"  n={n:<4} PR={pr:.2f}  (planted {K})")

    print("\ntest_analysis: PASS")


if __name__ == "__main__":
    main()
