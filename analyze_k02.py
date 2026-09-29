"""k02 analysis against experiments/PREREGISTER_k02_grid.md.

Criteria B1-B5 and Arm A were committed before these results existed. Anything not in that
file is exploratory and is labelled so.
"""
import json, numpy as np
from collections import defaultdict
from src.tasks.algebra import describe
from src.analysis.sparsity import gini, participation_ratio, key_freqs_5x_median
from src.analysis.transforms import unit_index
from src.analysis.stats import spearman

D = "results/k02_grid"
S = json.load(open(f"{D}/grid_summary.json"))
MOD = [113, 121, 125, 119, 120, 165]


def fnorms(W, n):
    k = np.arange(1, n // 2 + 1)[:, None]; t = np.arange(n)[None, :]
    s = np.sin(2 * np.pi * k * t / n) @ W; c = np.cos(2 * np.pi * k * t / n) @ W
    return np.sqrt((s ** 2).sum(1) + (c ** 2).sum(1))


def emb(arm, n, s):
    z = np.load(f"{D}/WE_{arm}_n{n}_s{s}.npz", allow_pickle=False)
    W = z["W_E"][:n].astype(float)
    return W - W.mean(0, keepdims=True)


print("=" * 78); print("ARM A — O1: does IPR_mult fall at 40k epochs, units only?"); print("=" * 78)
a = [r for r in S if r["arm"] == "A_replication"][0]
z = np.load(f"{D}/WE_A_replication_n113_s0.npz", allow_pickle=False)
W = z["W_E"][:113].astype(float)
from math import gcd
U = np.array([x for x in range(113) if gcd(x, 113) == 1])
idx, orders, _ = unit_index(113)
Ud = np.array(sorted(idx, key=lambda x: idx[x]))
Wu = W[Ud] - W[Ud].mean(0, keepdims=True)
fm = fnorms(Wu, len(Ud)); fa = fnorms(W[U] - W[U].mean(0, keepdims=True), len(U))
print(f"  grok step {a['grok_step']}, final acc {a['final_test_acc']:.4f}, 40k epochs, units only")
# Convention (open question O1; LAB_NOTEBOOK Entry 16). fnorms returns amplitude |A_k|.
# Gini is reported on amplitude: 2606.17399's 0.579 matches amplitude, not energy (energy
# inflates it to 0.90). The participation ratio is defined on normalised energy
# p_k = |A_k|^2 / sum|A_k|^2, which is what their 4.1 is (amplitude gives 12.8, energy
# 3.99). Feeding amplitude into the PR produced an "IPR_mult 12.5 vs 4.1" gap that was a
# convention error, not a circuit difference.
pr_energy = lambda f: participation_ratio(f ** 2)
print(f"  Gini_mult {gini(fm):.3f}   PR_mult {pr_energy(fm):.2f} (energy convention)   "
      f"key {[k+1 for k in key_freqs_5x_median(fm)]}")
print(f"  Gini_add  {gini(fa):.3f}   PR_add  {pr_energy(fa):.2f}")
print(f"      [amplitude convention, for the record: PR_mult {participation_ratio(fm):.2f}, "
      f"PR_add {participation_ratio(fa):.2f} -- NOT comparable to 2606.17399]")
print(f"  PUBLISHED (2606.17399): Gini_mult 0.579   PR_mult 4.1   Gini_add 0.071   PR_add 52.7")
print(f"  RESULT   : PR_mult = {pr_energy(fm):.2f} against published 4.1  -> "
      f"{'MATCH -- O1 RESOLVED' if abs(pr_energy(fm) - 4.1) < 1.0 else 'STILL DISCREPANT'}")

print("\n" + "=" * 78); print("ARM B — 5 seeds x 6 moduli"); print("=" * 78)
by = defaultdict(list)
for r in S:
    if r["arm"] == "B_thesis":
        by[r["n"]].append(r)

print("\nB1: does every modulus grok in >=4 of 5 seeds?")
print(f"  {'n':>5} {'grok steps (5 seeds)':>34} {'mean':>8} {'sd':>7} {'#grokked':>9}")
gs = {}
for n in MOD:
    rs = sorted(by[n], key=lambda r: r["seed"])
    g = [r["grok_step"] for r in rs]
    ok = [x for x in g if x is not None]
    gs[n] = ok
    print(f"  {n:>5} {str(g):>34} {np.mean(ok):8.0f} {np.std(ok):7.0f} {len(ok)}/5"
          + ("   <-- FAILS B1" if len(ok) < 4 else ""))

print("\nB3: is n=119 slowest by more than 1 SD from every other modulus?")
m119, s119 = np.mean(gs[119]), np.std(gs[119])
worst = max((np.mean(gs[n]) for n in MOD if n != 119))
print(f"  n=119 mean {m119:.0f} +- {s119:.0f}; next slowest mean {worst:.0f}")
print(f"  -> {'HELD' if m119 - worst > s119 else 'NOT HELD — 119 sits inside the spread'}")
print(f"  phi(n)-normalised (steps / phi(n)):")
for n in MOD:
    print(f"     n={n:<4} phi={describe(n)['phi']:<4} {np.mean(gs[n])/describe(n)['phi']:8.1f}")

print("\nB2: does the clock survive prime powers, over 5 seeds?")
print(f"  {'n':>5} {'Gini_mult mean+-sd':>22} {'Gini_add mean+-sd':>22} {'key_mult (seed0)':>20}")
gm = {}
for n in MOD:
    vals_m, vals_a = [], []
    for r in by[n]:
        W = emb("B_thesis", n, r["seed"])
        idx, orders, _ = unit_index(n)
        vals_a.append(gini(fnorms(W, n)))
        if len(orders) == 1:
            Ud = np.array(sorted(idx, key=lambda x: idx[x]))
            vals_m.append(gini(fnorms(W[Ud] - W[Ud].mean(0, keepdims=True), len(Ud))))
    gm[n] = vals_m
    k0 = ""
    if vals_m:
        W = emb("B_thesis", n, 0); idx, _, _ = unit_index(n)
        Ud = np.array(sorted(idx, key=lambda x: idx[x]))
        k0 = str([k + 1 for k in key_freqs_5x_median(fnorms(W[Ud]-W[Ud].mean(0,keepdims=True), len(Ud)))])
    ms = f"{np.mean(vals_m):.3f}+-{np.std(vals_m):.3f}" if vals_m else "non-cyclic"
    print(f"  {n:>5} {ms:>22} {np.mean(vals_a):.3f}+-{np.std(vals_a):.3f}      {k0:>20}")
if gm[113] and gm[121] and gm[125]:
    d121 = abs(np.mean(gm[121]) - np.mean(gm[113])); d125 = abs(np.mean(gm[125]) - np.mean(gm[113]))
    print(f"  |Gini_mult(121) - Gini_mult(113)| = {d121:.3f}   "
          f"|125 - 113| = {d125:.3f}   PREDICTED both < 0.10")
    print(f"  -> {'HELD' if max(d121,d125) < 0.10 else 'NOT HELD'}")

print("\nB5: does Gini_add track zero-divisor density across moduli (seed-averaged)?")

dens, ga = [], []
for n in MOD:
    d = describe(n)["zero_divisors"] / n
    v = np.mean([gini(fnorms(emb("B_thesis", n, r["seed"]), n)) for r in by[n]])
    dens.append(d); ga.append(v)
    print(f"  n={n:<4} density {d:.3f}  Gini_add {v:.3f}")
r = np.corrcoef(dens, ga)[0, 1]
rk = spearman(dens, ga)
print(f"  Pearson r = {r:.3f}   Spearman rho = {rk:.3f}   PREDICTED rho > 0.8")
print(f"  -> {'HELD' if rk > 0.8 else 'NOT HELD'}")

print("\nB4: CRT-dual law — permutation enrichment p<0.01 in >=4 of 5 seeds?")
import test_crt_law as T
print(f"  {'n':>5} {'enrichment per seed':>44} {'#p<0.01':>8}")
for n in [165, 120, 119]:
    es, ps = [], []
    for r in sorted(by[n], key=lambda r: r["seed"]):
        W = emb("B_thesis", n, r["seed"])
        e = T.freq_energy(W, n)
        res = T.permutation_test(e, [k - 1 for k in T.predicted(n)])
        es.append(res[0]); ps.append(res[1])
    ok = sum(p < 0.01 for p in ps)
    print(f"  {n:>5} {str([f'{x:.1f}x' for x in es]):>44} {ok}/5"
          + ("   <-- FAILS B4" if ok < 4 else ""))
print("  (113/121/125 are prime powers: VACUOUS by construction, not tested)")
