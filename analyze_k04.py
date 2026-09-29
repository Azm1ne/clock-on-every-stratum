"""k04 (N6) analysis -- implements experiments/PREREGISTER_k04_extended_moduli.md exactly.

Criteria 1-6 were committed before this run existed. Anything not in that file is
exploratory and is labelled so.

  P1  C19 survives 18 moduli      Mann-Whitney p < 0.05, cyclic vs non-cyclic steps/phi(n)
  P2  C9  survives 18 moduli      Spearman rho > 0.6, permutation p < 0.01
  P3  C8  holds on 9 fresh moduli >=2/3 seeds enriched at >=7 of 9 CRT-testable moduli
  P4  C6  extends to 49, 81, 169  |Gini_mult - 0.559| < 0.10 in >=2/3 seeds
  P5  dataset-size exclusion      a modulus grokking in <2/3 seeds is dropped from P1

k02 Arm B and k04 POOL (byte-identical protocol). Neither pools with the units-only
replication arm. Prime powers are VACUOUS for P3 by construction and never counted.

Runs partial when k04 has not landed: every criterion that can be computed from k02 alone
is reported and the rest say NOT LANDED. A gap in what was saved is never absence.
"""
import itertools, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.tasks.algebra import describe
from src.analysis.sparsity import gini
from src.analysis.transforms import unit_index
from src.analysis.stats import rankdata, perm_p
from src import provenance
import test_crt_law as T

K02, K04 = "results/k02_grid", "results/k04_extended"
K02_MOD = [113, 121, 125, 119, 120, 165]
K04_MOD = [49, 81, 169, 100, 54, 63, 75, 98, 99, 147, 105, 143]
NEW_PRIME_POWERS = [49, 81, 169]
K02_CYCLIC_BASELINE = 0.559          # mean Gini_mult of 113/121/125, from the ledger
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- loading

def summaries():
    """Arm B rows from both runs, tagged with their source directory."""
    rows = []
    for d, f in [(K02, "grid_summary.json"), (K04, "extended_summary.json")]:
        p = f"{d}/{f}"
        if not os.path.exists(p):
            print(f"  NOT LANDED: {p}")
            continue
        for r in json.load(open(p)):
            if r["arm"] == "B_thesis":
                rows.append(dict(r, dir=d))
    return rows


def emb(d, n, s):
    """Mean-centred embedding rows for the n residues."""
    W = np.load(f"{d}/WE_B_thesis_n{n}_s{s}.npz", allow_pickle=False)["W_E"][:n].astype(float)
    return W - W.mean(0, keepdims=True)


def gini_add(d, n, s):
    return gini(T.freq_energy(emb(d, n, s), n))


def gini_mult(d, n, s):
    """Discrete-log reordering, then the additive DFT over phi(n) points. Cyclic only."""
    idx, orders, _ = unit_index(n)
    if len(orders) != 1:
        return None
    Ud = np.array(sorted(idx, key=lambda x: idx[x]))
    W = np.load(f"{d}/WE_B_thesis_n{n}_s{s}.npz", allow_pickle=False)["W_E"][:n].astype(float)
    Wu = W[Ud] - W[Ud].mean(0, keepdims=True)
    return gini(T.freq_energy(Wu, len(Ud)))


# ---------------------------------------------------------------- statistics

def mann_whitney(x, y):
    """U statistic for x vs y, and an EXACT two-sided permutation p over label
    assignments. No scipy: C(18,8) = 43,758 assignments is cheap to enumerate, and an
    exact p needs no normal approximation at n=8 vs 10."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    pool = np.concatenate([x, y])
    m = len(x)

    def U(sel):
        a, b = pool[list(sel)], np.delete(pool, list(sel))
        return float((np.sign(a[:, None] - b[None, :]) + 1).sum() / 2)   # ties count 1/2

    obs = U(range(m))
    idx = range(len(pool))
    null = np.array([U(c) for c in itertools.combinations(idx, m)])
    centre = m * len(y) / 2
    p = float((np.abs(null - centre) >= abs(obs - centre) - 1e-9).mean())
    return obs, p, len(null)


def spearman_perm(a, b, n_perm=20_000):
    ra, rb = rankdata(a), rankdata(b)          # MIDRANKS -- zdd ties at n=63/147
    rho = float(np.corrcoef(ra, rb)[0, 1])
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.corrcoef(ra, RNG.permutation(rb))[0, 1]
    return rho, perm_p(int((null >= rho).sum()), n_perm)


# ---------------------------------------------------------------- criteria

def per_modulus(rows):
    """One row per modulus: grok rate, mean steps, phi, cyclicity, density."""
    out = {}
    for n in sorted({r["n"] for r in rows}):
        rs = [r for r in rows if r["n"] == n]
        g = [r["grok_step"] for r in rs]
        ok = [x for x in g if x is not None]
        d = describe(n)
        out[n] = dict(n=n, dir=rs[0]["dir"], seeds=[r["seed"] for r in rs],
                      n_seeds=len(rs), n_grokked=len(ok),
                      grok_steps=g, mean_steps=float(np.mean(ok)) if ok else None,
                      phi=d["phi"], cyclic=d["cyclic"],
                      density=d["zero_divisors"] / n,
                      squarefree=d["squarefree"])
    return out


def p5_exclusions(mods):
    """Pre-registered: grok in <2/3 of seeds -> dropped from P1, reported as a
    dataset-size null, never as an algebraic effect."""
    return [m for m in mods.values() if m["n_grokked"] / m["n_seeds"] < 2 / 3]


def p1(mods, excluded, label="all 18"):
    keep = [m for m in mods.values() if m not in excluded and m["mean_steps"]]
    cyc = [m["mean_steps"] / m["phi"] for m in keep if m["cyclic"]]
    non = [m["mean_steps"] / m["phi"] for m in keep if not m["cyclic"]]
    if len(cyc) < 2 or len(non) < 2:
        print(f"  {label}: too few moduli ({len(cyc)} cyclic, {len(non)} non-cyclic)")
        return None
    U, p, nperm = mann_whitney(cyc, non)
    med_c, med_n = float(np.median(cyc)), float(np.median(non))
    print(f"  {label:<22} cyclic n={len(cyc)} median {med_c:7.1f}   "
          f"non-cyclic n={len(non)} median {med_n:7.1f}")
    print(f"  {'':<22} U={U:.1f}  exact permutation p={p:.4f} over {nperm:,} assignments")
    held = med_c < med_n and p < 0.05
    print(f"  {'':<22} -> {'HELD' if held else 'NOT HELD'}"
          + ("" if held else "   (C19 is RETRACTED if this is the pre-registered row)"))
    return held


def main():
    print("=" * 78)
    print("k04 -- the extended modulus set, pooled with k02 Arm B")
    print("=" * 78)
    rows = summaries()
    if not rows:
        sys.exit("no Arm B results at all")
    mods = per_modulus(rows)
    landed = sorted(set(mods) & set(K04_MOD))
    print(f"\nmoduli present: {len(mods)}  (k02 {sorted(set(mods) & set(K02_MOD))}, "
          f"k04 {landed}{' -- NOT LANDED' if not landed else ''})")

    print("\n" + "-" * 78)
    print("P5 -- grok rate per modulus, and the pre-registered exclusion")
    print("-" * 78)
    print(f"  {'n':>5} {'phi':>5} {'cyc':>4} {'dens':>6} {'grokked':>8} {'mean steps':>11} "
          f"{'steps/phi':>10}")
    for n, m in sorted(mods.items()):
        ms = f"{m['mean_steps']:.0f}" if m["mean_steps"] else "--"
        sp = f"{m['mean_steps']/m['phi']:.1f}" if m["mean_steps"] else "--"
        print(f"  {n:>5} {m['phi']:>5} {'Y' if m['cyclic'] else 'n':>4} "
              f"{m['density']:>6.3f} {m['n_grokked']}/{m['n_seeds']:<6} {ms:>11} {sp:>10}")
    ex = p5_exclusions(mods)
    print(f"  excluded from P1 (grok < 2/3 seeds): "
          f"{[m['n'] for m in ex] if ex else 'none'}  -- dataset-size null, not algebra")

    print("\n" + "-" * 78)
    print("P1 -- C19: does phi(n)-normalised grokking time separate cyclic from non-cyclic?")
    print("-" * 78)
    p1(mods, ex, "PRE-REGISTERED")
    # pre-registered sensitivity check: the two smallest-phi moduli are extreme points
    small = sorted(mods.values(), key=lambda m: m["phi"])[:2]
    print(f"  sensitivity (stated in the pre-registration, NOT the criterion): "
          f"dropping the two smallest phi {[m['n'] for m in small]}")
    p1(mods, ex + small, "minus smallest phi")

    # EXPLORATORY, not pre-registered. P1's normalisation divides by phi(n), and in k02
    # the three cyclic moduli happened to be the three largest phi (112, 110, 100) while
    # the non-cyclic ones were 96, 32, 80. If steps/phi(n) is itself a function of phi,
    # C19's "separation" was reading phi, not cyclicity -- the same size confound that
    # retracted C7b and C20. Cheap to check, so check it.
    print("\n" + "-" * 78)
    print("EXPLORATORY (not pre-registered) -- is steps/phi(n) just a function of phi(n)?")
    print("-" * 78)
    keep = [m for m in mods.values() if m not in ex and m["mean_steps"]]
    ph = np.array([m["phi"] for m in keep], float)
    sp = np.array([m["mean_steps"] / m["phi"] for m in keep])
    raw = np.array([m["mean_steps"] for m in keep], float)
    r1, p1_ = spearman_perm(-ph, sp)
    r2, p2_ = spearman_perm(ph, raw)
    print(f"  Spearman rho(-phi, steps/phi) = {r1:+.3f}  perm p = {p1_:.2e}   "
          f"(n={len(keep)} grokking moduli)")
    print(f"  Spearman rho( phi, raw steps) = {r2:+.3f}  perm p = {p2_:.2e}")
    cyc = [m for m in keep if m["cyclic"]]
    non = [m for m in keep if not m["cyclic"]]
    print(f"  median phi: cyclic {np.median([m['phi'] for m in cyc]):.0f}  "
          f"non-cyclic {np.median([m['phi'] for m in non]):.0f}")
    print("  -> if rho(-phi, steps/phi) is strong, phi(n) and not cyclicity is what the")
    print("     k02 ordering was reading. C19 would be a normalisation artefact.")

    print("\n" + "-" * 78)
    print("P2 -- C9: does Gini_add track zero-divisor density over 18 moduli?")
    print("-" * 78)
    dens, ga = [], []
    for n, m in sorted(mods.items()):
        v = float(np.mean([gini_add(m["dir"], n, s) for s in m["seeds"]]))
        dens.append(m["density"]); ga.append(v)
        print(f"  n={n:<4} density {m['density']:.3f}  Gini_add {v:.3f}")
    rho, p = spearman_perm(np.array(dens), np.array(ga))
    print(f"  Spearman rho = {rho:.3f}   permutation p = {p:.2e}   "
          f"PREDICTED rho > 0.6 and p < 0.01")
    print(f"  -> {'HELD' if rho > 0.6 and p < 0.01 else 'NOT HELD'}")

    print("\n" + "-" * 78)
    print("P3 -- C8: the CRT-dual law on the 9 fresh CRT-testable moduli")
    print("-" * 78)
    testable = [n for n in landed if n not in NEW_PRIME_POWERS]
    if not testable:
        print("  NOT LANDED -- k04 has produced no CRT-testable modulus yet")
    else:
        hold = 0
        print(f"  {'n':>5} {'enrichment per seed':>34} {'#p<0.01':>8}")
        for n in testable:
            m = mods[n]
            es, ps = [], []
            for s in m["seeds"]:
                e = T.freq_energy(emb(m["dir"], n, s), n)
                r = T.permutation_test(e, [k - 1 for k in T.predicted(n)])
                if r is None:
                    continue
                es.append(r[0]); ps.append(r[1])
            ok = sum(x < 0.01 for x in ps)
            hold += ok >= 2
            print(f"  {n:>5} {str([f'{x:.1f}x' for x in es]):>34} {ok}/{len(ps)}"
                  + ("" if ok >= 2 else "   <-- does not hold here"))
        print(f"  holds at {hold} of {len(testable)} CRT-testable moduli; "
              f"PREDICTED >=7 of 9")
        print(f"  -> {'HELD' if hold >= 7 else 'NOT HELD'}")
        print(f"  (prime powers {NEW_PRIME_POWERS} are VACUOUS by construction, not tested)")

    print("\n" + "-" * 78)
    print("P4 -- C6: does the clock survive at 49 = 7^2, 81 = 3^4, 169 = 13^2?")
    print("-" * 78)
    got = [n for n in NEW_PRIME_POWERS if n in mods]
    if not got:
        print("  NOT LANDED -- none of 49, 81, 169 has produced a checkpoint yet")
    else:
        allok = True
        print(f"  {'n':>5} {'Gini_mult per seed':>34} {'|delta| vs 0.559':>18} {'within 0.10':>12}")
        for n in got:
            m = mods[n]
            vals = [gini_mult(m["dir"], n, s) for s in m["seeds"]]
            vals = [v for v in vals if v is not None]
            dl = [abs(v - K02_CYCLIC_BASELINE) for v in vals]
            ok = sum(d < 0.10 for d in dl)
            allok &= ok >= 2
            print(f"  {n:>5} {str([f'{v:.3f}' for v in vals]):>34} "
                  f"{str([f'{d:.3f}' for d in dl]):>18} {ok}/{len(dl):<12}"
                  + ("" if ok >= 2 else "   <-- C6 does NOT extend here"))
        print(f"  -> {'HELD' if allok else 'NOT HELD'}")

    print("\n" + "-" * 78)
    print("Criterion 6 -- provenance: every .npz carries a REAL git SHA")
    print("-" * 78)
    bad = []
    for n, m in sorted(mods.items()):
        for s in m["seeds"]:
            f = f"{m['dir']}/WE_B_thesis_n{n}_s{s}.npz"
            try:
                sha = (provenance.read(f) or {}).get("git_sha", "unknown")
            except Exception:
                sha = "unknown"
            if sha in ("unknown", "", None) or sha.endswith("-dirty"):
                bad.append((f, sha))
    print(f"  {len(bad)} of {sum(m['n_seeds'] for m in mods.values())} artifacts lack a "
          f"clean SHA")
    for f, sha in bad[:6]:
        print(f"    {f}  git_sha={sha}")
    if len(bad) > 6:
        print(f"    ... and {len(bad)-6} more")
    print("  (the 31 k02 runs predate the stamping line -- recorded in STATE.md, "
          "not repairable without re-running)")


if __name__ == "__main__":
    main()
