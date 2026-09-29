"""N4 / N4b -- restricted / excluded loss for `a*b mod n`. Causal, not just descriptive.

Protocol invariant 5: "every 'X is responsible' claim needs an ablation, not just a
sparsity number." C6 (the clock survives prime powers) had rested on sparsity alone.
This is the causal test.

Why exponent coordinates: Nanda's restricted set puts f(a+b) at DFT bins (k,k). For
`a*b mod n` that is only true after relabelling units by their discrete log, where
multiplication becomes addition (2606.17399 eq. 2). Ablating (k,k) on the raw residue
grid tests a hypothesis no one holds; in raw coordinates neuron_term_decomposition reports
"a+b 0.4%" for a model that is 100% a+b in dlog coordinates (Entry 16).

N4b (this version): a non-cyclic unit group has no discrete log, but it still has a
canonical exponent coordinate. (Z/nZ)* = Z_o1 x ... x Z_or, every unit is uniquely
prod g_i^e_i, and multiplication is still addition of exponent tuples. A character is
then a multi-index and the logit grid is a 2r-dimensional array. `src/analysis/ablation`
takes `n` either as a scalar or as the tuple of component orders, so 113 (r=1) and 119
(r=2) run through the same code, which is what makes their numbers comparable.

Zero divisors have no exponent coordinate at all and are excluded from the grid; that
loss is itself the object of study, and it is reported, not hidden.
"""
import json, sys, os
import numpy as np
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.analysis.ablation import (ablation_report, key_freq_pairs, fourier_ablate,
                                   fourier_spectrum,
                                   cross_entropy_np, all_freq_labels,
                                   unit_logit_grid)
from src.analysis.transforms import unit_index, character_transform, energy
from src.analysis.sparsity import key_freqs_5x_median
from src import provenance

D = sys.argv[1] if len(sys.argv) > 1 else "results/k02_grid"
N_CONTROL = 10000        # R1 (PREREGISTER_r1_permutation_B.md). 20 was far too few for a
                         # heavy-tailed control; 200 cannot express a p below 1/201.


def generates_dual(key, orders):
    """Does the key set generate the character group?

    If it does not, no clock built from those characters can be exact: the restricted
    circuit cannot separate two units that every key character maps to the same value.
    Found while building the planted-signal test -- [(0,2),(1,1)] over Z_4 x Z_6 tops out
    at 50% accuracy. Reported as a diagnostic, not a criterion.
    """
    o = np.array(orders)
    seen = {tuple([0] * len(orders))}
    frontier = list(seen)
    while frontier:
        nxt = []
        for x in frontier:
            for k in key:
                y = tuple((np.array(x) + np.array(k)) % o)
                if y not in seen:
                    seen.add(y); nxt.append(y)
        frontier = nxt
    return len(seen), int(np.prod(o))


def run(tag, n, powers=None, verbose=True, d=None, return_draws=False):
    # `d` and `return_draws` exist so a figure can plot the permutation null from THIS
    # implementation instead of a second copy of it. Defaults preserve the old behaviour.
    f = f"{d or D}/WE_{tag}.npz"
    if not os.path.exists(f):
        print(f"  {tag}: no checkpoint"); return None
    z = np.load(f, allow_pickle=False)
    if "logits_all" not in z.files:
        print(f"  {tag}: logits_all NOT SAVED -- cannot ablate"); return None

    # `powers` varies the choice of generator per cyclic component. Nothing claimed about
    # a circuit may depend on it, so the equivariance check threads it through this path
    # rather than a parallel one (test_generator_equivariance.py).
    # Grid assembly, the key set and the exponent coordinate all live in
    # src.analysis.ablation.unit_logit_grid so that the figure which draws this
    # spectrum and the test which ablates it cannot disagree about either.
    # energy() folds conjugates and drops nothing, so its index is the label index once
    # DC is kept (no +1). Key frequencies go on amplitude, protocol invariant 2.
    G, y, orders, kf = unit_logit_grid(z, n, powers)
    q = int(np.prod(orders))                      # phi(n)
    if not kf:
        print(f"  {tag}: no key characters above 5x median -- nothing to ablate")
        return None

    r = ablation_report(G, y, kf, orders)
    pool = all_freq_labels(orders)
    # Control: random equal-sized character sets. Without it "removing these hurts" is not
    # evidence, since removing any 4 characters removes energy.
    #
    # The statistic is a permutation p, not a ratio to the control mean. The control
    # distribution is bimodal and heavy-tailed: ~73% of random removals leave the loss
    # untouched (the model does not use those characters) and the rest are catastrophic,
    # so its mean depends on which few catastrophic draws land. At n=121 seed 0,
    # excluded/mean(control) ranges over 21x-440x across control RNG seeds alone, on
    # identical data. The fraction of draws at least as damaging as the key set is stable,
    # and is the same threshold-free permutation logic C8 uses (test_crt_law.py).
    rng = np.random.default_rng(0)
    ctrl = []
    # G does not change across the null, so its forward transform is computed once. Same
    # arithmetic, asserted bit-identical in test_ablation.py.
    F = fourier_spectrum(G, len(orders))
    for _ in range(N_CONTROL):
        rk = [pool[i] for i in rng.choice(len(pool), size=len(kf), replace=False)]
        ctrl.append(cross_entropy_np(
            fourier_ablate(G, remove=key_freq_pairs(rk, orders),
                           spectrum=F).reshape(q * q, -1), y))
    ctrl = np.array(ctrl)
    # PHIPSON-SMYTH, PRE-REGISTERED IN R1: p_hat = (1+b)/(B+1), never b/B. A Monte-Carlo
    # p cannot claim more resolution than its draw count, and b/B says p = 0 exactly --
    # which the paper then printed to four decimals in eight places.
    n_ge = int((ctrl >= r["excluded"]).sum())
    p_perm = (1.0 + n_ge) / (len(ctrl) + 1.0)
    sep_med = r["excluded"] / max(float(np.median(ctrl)), 1e-30)

    grp = "x".join(f"Z_{o}" for o in orders)
    gen, full = generates_dual(kf, orders)
    sep = r['excluded'] / max(ctrl.mean(), 1e-30)
    row = dict(n=n, tag=tag, orders=orders, cyclic=len(orders) == 1,
               n_draws=int(len(ctrl)), n_ge=n_ge,
               n_key=len(kf), key=sorted(kf), generates=(gen == full),
               **{k: v for k, v in r.items() if k != 'per_freq'},
               control_mean=float(ctrl.mean()), control_median=float(np.median(ctrl)),
               p_perm=p_perm, sep_median=float(sep_med), sep_mean_UNSTABLE=float(sep))
    if return_draws:
        return row, ctrl
    if not verbose:
        return row
    print(f"  {tag}  n={n}  (Z/nZ)* = {grp}  phi={q}")
    print(f"      key characters {kf}")
    print(f"      key set generates {gen}/{full} of the character group"
          f"{'' if gen == full else '   <-- CANNOT be an exact clock on its own'}")
    print(f"      baseline loss          {r['baseline']:.6e}")
    print(f"      RESTRICTED (keep only) {r['restricted']:.6e}   "
          f"{'BETTER' if r['restricted'] < r['baseline'] else 'worse'} "
          f"({r['baseline']/max(r['restricted'],1e-30):.2f}x)")
    print(f"      EXCLUDED   (remove)    {r['excluded']:.6e}   "
          f"{r['excluded']/max(r['baseline'],1e-30):.3e}x baseline")
    print(f"      control: remove {len(kf)} RANDOM characters, {N_CONTROL} draws -> "
          f"median {np.median(ctrl):.6e}  mean {ctrl.mean():.6e} (mean is UNSTABLE)")
    print(f"      permutation p (draws >= excluded) = {p_perm:.4f}"
          f"   ({n_ge} of {len(ctrl)} draws; floor 1/(B+1) = {1/(len(ctrl)+1):.2e})")
    print(f"      excluded / MEDIAN control = {sep_med:.3e}x   "
          f"-> {'KEY CHARACTERS ARE CAUSALLY RESPONSIBLE' if p_perm < 0.01 else 'NOT SEPARATED'}")
    top = sorted(r['per_freq'].items(), key=lambda kv: -kv[1])[:6]
    print(f"      top per-character ablation deltas: "
          + ", ".join(f"k={k} {d:+.3f}" for k, d in top))
    print(f"      key set by ablation rank: {sorted([k for k, _ in top[:len(kf)]])}  "
          f"vs by embedding norm: {sorted(kf)}")
    return row


if __name__ == "__main__":
    import glob, re
    print("N4/N4b -- restricted/excluded ablation for a*b mod n, exponent coordinates\n")
    # Parse the tag first, then find the modulus INSIDE it. The old pattern demanded
    # `_n<digits>_s<digits>` adjacent, so k05's `B_thesis_n49_f80_s0` matched nothing and
    # the script printed an empty table rather than saying it had found no runs.
    tags = []
    skipped = []
    for f in sorted(glob.glob(f"{D}/WE_*.npz")):
        base = os.path.basename(f)
        m = re.match(r"WE_(.+)\.npz$", base)
        n = re.search(r"_n(\d+)(?:_|$)", m.group(1)) if m else None
        if m and n:
            tags.append((m.group(1), int(n.group(1))))
        else:
            skipped.append(base)
    if skipped:
        print(f"  {len(skipped)} file(s) whose modulus could not be parsed, SKIPPED: "
              f"{skipped[:3]}{' ...' if len(skipped) > 3 else ''}\n")
    if not tags:
        sys.exit(f"no parsable WE_*.npz in {D} -- refusing to print an empty table")
    rows = []
    for tag, n in tags:
        r = run(tag, n)
        if r: rows.append(r)
        print()

    # A number with no code version attached is not usable.
    out = f"{D.rstrip('/').split('/')[-1]}_n4_ablation.npz"
    out = f"results/{out}"
    np.savez(out, **provenance.stamp_npz(dict(source=D, n_runs=len(rows))),
             rows=np.array([json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                        for k, v in r.items()}) for r in rows]))
    print(f"saved {out}\n")

    print("=" * 86)
    print(f"{'run':<28} {'group':>14} {'baseline':>11} {'restricted':>11} "
          f"{'excluded':>11} {'perm p':>8}")
    for r in rows:
        grp = "x".join(f"Z_{o}" for o in r["orders"])
        print(f"{r['tag']:<28} {grp:>14} {r['baseline']:>11.3e} {r['restricted']:>11.3e} "
              f"{r['excluded']:>11.3e} {r['p_perm']:>8.4f}")

    # Seed aggregate: C22/C23 rested on seed 0 alone until k03 saved logits_all for all
    # five. The separation from the random-character control is the criterion.
    print("\n" + "=" * 86)
    print("PER-MODULUS, OVER SEEDS  (arm B only)")
    print(f"{'n':>5} {'group':>16} {'seeds':>6} {'restricted/baseline':>20} "
          f"{'max perm p':>11} {'p<0.01 in':>10}")
    for n in sorted({r["n"] for r in rows if r["tag"].startswith("B_")}):
        rs = [r for r in rows if r["n"] == n and r["tag"].startswith("B_")]
        imp = np.array([r["baseline"] / max(r["restricted"], 1e-30) for r in rs])
        pp = np.array([r["p_perm"] for r in rs])
        grp = "x".join(f"Z_{o}" for o in rs[0]["orders"])
        print(f"{n:>5} {grp:>16} {len(rs):>6} "
              f"{f'{imp.mean():.2f}x +/- {imp.std():.2f}':>20} "
              f"{pp.max():>11.4f} {f'{int((pp < 0.01).sum())}/{len(rs)}':>10}")
