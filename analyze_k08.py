"""k08 (O1b) analysis -- implements experiments/PREREGISTER_k08_init.md exactly.

Criteria 1-4 were committed before this run existed. Anything not in that file is
exploratory and is labelled so.

  I1  the ordering        median upstream_heavy < fanin < downstream_heavy, exact MW p<0.05
  I2  the magnitude       downstream_heavy median >= 1.4 x fanin median
  I3  the same circuit    |dGini_mult| < 0.10 and C22 p < 0.01 in >=6/8
  I4  grok rate           >=6/8 per condition, else censored and dropped from I1

The modulus is FIXED at 113, so the phi(n) confound that retracted C19 cannot operate here.
This is the one clean grokking-time comparison in the project -- and per the
pre-registration, a difference could still be a WEIGHT-DECAY effect rather than an
initialisation effect, because wd = 1.0 is large and is not scale-invariant.
"""
import itertools, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.analysis.sparsity import gini
from src.analysis.transforms import unit_index
from src import provenance
import test_crt_law as T

D = sys.argv[1] if len(sys.argv) > 1 else "results/k08_init"
N = 113
CONDS = ["upstream_heavy", "fanin", "downstream_heavy"]
SCALE = {"upstream_heavy": 0.25, "fanin": 1.0, "downstream_heavy": 4.0}
MIN_GROK = 6            # of 8, pre-registered
GAP = 1.4               # the real O1b discrepancy is 1.4x-2.2x


def path(c, s):
    return f"{D}/WE_{c}_n{N}_s{s}.npz"


def gini_mult(c, s):
    if not os.path.exists(path(c, s)):
        return None
    idx, orders, _ = unit_index(N)
    Ud = np.array(sorted(idx, key=lambda x: idx[x]))
    W = np.load(path(c, s), allow_pickle=False)["W_E"][:N].astype(float)
    Wu = W[Ud] - W[Ud].mean(0, keepdims=True)
    return gini(T.freq_energy(Wu, len(Ud)))


def mann_whitney(x, y):
    """U and an EXACT two-sided permutation p. 8 vs 8 = 12,870 assignments -- enumerable,
    and the normal approximation is not credible on a distribution whose spread is 3.8x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    pool = np.concatenate([x, y]); m = len(x)
    U = lambda sel: float((np.sign(pool[list(sel)][:, None]
                                   - np.delete(pool, list(sel))[None, :]) + 1).sum() / 2)
    obs = U(range(m))
    null = np.array([U(c) for c in itertools.combinations(range(len(pool)), m)])
    centre = m * len(y) / 2
    return obs, float((np.abs(null - centre) >= abs(obs - centre) - 1e-9).mean()), len(null)


def main():
    p = f"{D}/init_summary.json"
    if not os.path.exists(p):
        sys.exit(f"NOT LANDED: {p}")
    rows = json.load(open(p))
    print("=" * 78)
    print("k08 (O1b) -- does initialisation scale set the grokking time?")
    print("=" * 78)
    print(f"\n{len(rows)} runs present of the 24 pre-registered")
    print("  PREMISE CORRECTED IN THE PRE-REGISTRATION: our init is plain fan-in scaling")
    print("  on every matrix (n_heads*d_head = d_model), so the asymmetry is IMPOSED here,")
    print("  not inherited.")

    by = {c: sorted([r for r in rows if r["arm"] == c], key=lambda r: r["seed"])
          for c in CONDS}

    print("\n" + "-" * 78)
    print("I4 / criterion 4 -- grok rate per condition")
    print("-" * 78)
    usable = []
    for c in CONDS:
        gs = [r["grok_step"] for r in by[c]]
        ok = [g for g in gs if g is not None]
        keep = len(ok) >= MIN_GROK
        usable.append(keep)
        print(f"  {c:<18} ds={SCALE[c]:<5} grokked {len(ok)}/{len(gs)}   steps {sorted(ok)}")
        if not keep:
            print(f"     <-- CENSORED: <{MIN_GROK}/8, dropped from I1 per the pre-registration")
    print(f"  -> I4 {'HELD' if all(usable) else 'NOT HELD'}")

    print("\n" + "-" * 78)
    print("I1 / criterion 1 -- the ordering")
    print("-" * 78)
    med = {}
    for c in CONDS:
        ok = [g for g in (r["grok_step"] for r in by[c]) if g is not None]
        med[c] = float(np.median(ok)) if ok else None
        if ok:
            print(f"  {c:<18} median {med[c]:>8.0f}   mean {np.mean(ok):>8.0f} "
                  f"+/- {np.std(ok):>6.0f}   n={len(ok)}")
    ordered = (med["upstream_heavy"] is not None and med["downstream_heavy"] is not None
               and med["fanin"] is not None
               and med["upstream_heavy"] < med["fanin"] < med["downstream_heavy"])
    print(f"  predicted ordering upstream_heavy < fanin < downstream_heavy: "
          f"{'YES' if ordered else 'NO'}")
    a = [g for g in (r["grok_step"] for r in by["upstream_heavy"]) if g is not None]
    b = [g for g in (r["grok_step"] for r in by["downstream_heavy"]) if g is not None]
    i1 = False
    if len(a) >= 2 and len(b) >= 2 and usable[0] and usable[2]:
        U, pv, nperm = mann_whitney(a, b)
        print(f"  extreme pair: U={U:.1f}  exact permutation p={pv:.4f} "
              f"over {nperm:,} assignments   (predicted < 0.05)")
        i1 = ordered and pv < 0.05
    else:
        print("  extreme pair not testable (a condition was censored)")
    print(f"  -> I1 {'HELD' if i1 else 'NOT HELD'}")
    if not i1:
        print("     Initialisation scale is NOT what makes us faster than the published")
        print("     runs. O1b stays open with its most plausible mechanism eliminated --")
        print("     a useful negative, and it is recorded as one.")

    print("\n" + "-" * 78)
    print("I2 / criterion 2 -- is the effect big enough to explain O1b at all?")
    print("-" * 78)
    print("  the gap to explain is 6,400 -> 9k-14k, i.e. 1.4x-2.2x")
    if med.get("fanin"):
        ratio = med["downstream_heavy"] / med["fanin"] if med.get("downstream_heavy") else float("nan")
        print(f"  downstream_heavy / fanin median ratio = {ratio:.2f}x   (predicted >= {GAP})")
        i2 = ratio >= GAP
        print(f"  -> I2 {'HELD' if i2 else 'NOT HELD'}")
        if not i2:
            print("     The effect exists but is too small to account for O1b. Say that")
            print("     plainly; do not report it as support.")

    print("\n" + "-" * 78)
    print("I3 / criterion 3 -- same circuit, or a different one?")
    print("-" * 78)
    base = [v for v in (gini_mult("fanin", r["seed"]) for r in by["fanin"]) if v is not None]
    bm = float(np.mean(base)) if base else None
    print(f"  fanin Gini_mult mean {bm:.3f}" if bm else "  fanin: no checkpoints")
    i3 = True
    for c in CONDS:
        vals = [v for v in (gini_mult(c, r["seed"]) for r in by[c]) if v is not None]
        if not vals or bm is None:
            continue
        d = abs(float(np.mean(vals)) - bm)
        ok = d < 0.10
        i3 &= ok
        print(f"  {c:<18} Gini_mult {np.mean(vals):.3f} +/- {np.std(vals):.3f}   "
              f"|delta| {d:.3f}   {'within 0.10' if ok else 'OUTSIDE 0.10'}")
    print(f"  -> I3 (sparsity half) {'HELD' if i3 else 'NOT HELD'}")
    print("  (the causal half -- C22 permutation p < 0.01 in >=6/8 -- runs through")
    print("   analyze_n4.py on this directory)")
    print("  This is the control that separates 'init changes WHEN the circuit forms'")
    print("  from 'init changes WHICH circuit forms'. Only the first is a timing result.")

    print("\n" + "-" * 78)
    print("criterion 5 -- provenance")
    print("-" * 78)
    seen = {}
    for r in rows:
        pr = provenance.read(path(r["arm"], r["seed"])) or {}
        k = (pr.get("git_sha", "unknown")[:7], pr.get("kaggle_account", "unknown"))
        seen[k] = seen.get(k, 0) + 1
    for (sha, acct), c in sorted(seen.items()):
        bad = sha in ("unknown", "") or sha.startswith("__")
        print(f"  git_sha {sha}  account {acct}  x{c}" + ("   <-- NOT REPRODUCIBLE" if bad else ""))

    print("\n  CONFOUND, stated in the pre-registration and not resolved by this run:")
    print("  weight decay is 1.0 and is NOT scale-invariant, so a difference here could be")
    print("  a decay effect rather than an initialisation effect. Separating them needs a")
    print("  wd sweep that k08 does not do.")


if __name__ == "__main__":
    main()
