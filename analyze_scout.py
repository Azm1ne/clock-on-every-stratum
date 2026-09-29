"""Scout analysis: does the basis-matching claim hold, and what happens at prime powers?

Zero GPU. Protocol per 2606.17399: transform the embedding, measure Gini on the
amplitude spectrum (DC dropped) and PR on energy (an earlier version put energy into the
Gini; corrections P6, P6b). Control: a random orthogonal basis on the same row subset the
character transform uses, so the comparison is dimension-matched.
"""
import numpy as np
from src.tasks.algebra import describe, units
from src.analysis.transforms import (additive_dft, character_transform,
                                     random_orthogonal, energy)
from src.analysis.sparsity import gini, participation_ratio, n_key_frequencies
from analyze_n7 import mult_amplitude, add_amplitude, rand_amplitude

MODULI = [113, 165, 119, 121, 125, 120]
print(f"{'n':>4} {'nonreg':>7} {'grok':>6} | {'Gini_mult':>9} {'Gini_add':>8} {'Gini_rnd':>8} "
      f"{'Gini_rndU':>9} | {'key_mult':>8} {'key_add':>7} | {'PR_mult':>7}")
print("-" * 104)
rows = []
for n in MODULI:
    d = describe(n)
    z = np.load(f"results/k01_scout/scout_n{n}_seed0.npz")
    W = z["W_E"][:n]                      # drop the "=" token row
    W = W - W.mean(0, keepdims=True)      # centre: the DC term is not a "frequency"
    U = np.array(units(n))

    e_mult = energy(character_transform(W, n))
    e_add  = energy(additive_dft(W))
    e_rnd  = energy(random_orthogonal(W, seed=0))

    r = dict(n=n, nonreg=d["n_nonregular"],
             grok=int(z["hist"][:, list(z["hist_cols"]).index("step")][
                 np.argmax(z["hist"][:, list(z["hist_cols"]).index("test_acc")] > 0.99)]),
             gm=gini(mult_amplitude(W, n)), ga=gini(add_amplitude(W, n)),
             gr=gini(np.sqrt(e_rnd)), gru=gini(rand_amplitude(W, n)),
             km=n_key_frequencies(e_mult), ka=n_key_frequencies(e_add),
             pr=participation_ratio(e_mult))
    rows.append(r)
    print(f"{n:>4} {r['nonreg']:>7} {r['grok']:>6} | {r['gm']:>9.3f} {r['ga']:>8.3f} "
          f"{r['gr']:>8.3f} {r['gru']:>9.3f} | {r['km']:>8} {r['ka']:>7} | {r['pr']:>7.2f}")

print("\nPRE-REGISTERED target (2606.17399, n=113): Gini_mult=0.58, Gini_add=0.07, 4 key freqs")
b = rows[0]
print(f"OURS               (n=113): Gini_mult={b['gm']:.2f}, Gini_add={b['ga']:.2f}, "
      f"{b['km']} key freqs")

# ---- follow-up: the additive column varies hugely. What governs it? ----
print("\nadditive-basis sparsity vs zero-divisor density:")
print(f"{'n':>4} {'nonunits':>9} {'density':>8} {'Gini_add':>9} {'key_add':>8}")
dens, ga = [], []
for r in sorted(rows, key=lambda r: describe(r["n"])["zero_divisors"] / r["n"]):
    n = r["n"]; zd = describe(n)["zero_divisors"]; f = zd / n
    dens.append(f); ga.append(r["ga"])
    print(f"{n:>4} {zd:>9} {f:>8.3f} {r['ga']:>9.3f} {r['ka']:>8}")
print(f"\nPearson r(zero-divisor density, Gini_add) = {np.corrcoef(dens, ga)[0,1]:.3f}")
print(f"Spearman-ish (rank) monotone: {np.all(np.diff(np.argsort(np.argsort(ga)))>=-1)}")

# J-classes are unions of arithmetic progressions (common difference d), which is
# exactly the structure the ADDITIVE DFT is sparse in. Test that directly: build the
# J-class indicator functions and measure their additive sparsity.
print("\nJ-class indicator functions, additive-basis sparsity (structure, not learned):")
for n in [113, 121, 165, 120]:
    ind = np.zeros((n, len(describe(n)["jclasses"])))
    for j, c in enumerate(describe(n)["jclasses"]):
        from math import gcd
        for x in range(n):
            if gcd(x, n) == c["d"]:
                ind[x, j] = 1
    ind -= ind.mean(0, keepdims=True)
    print(f"  n={n:<4} Gini_add(J-class indicators) = {gini(add_amplitude(ind, n)):.3f}")
