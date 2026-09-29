"""R2 -- is it the same clock on every stratum, or one clock each?

Scores the criteria pre-registered in `experiments/PREREGISTER_r2_cross_stratum.md`,
committed at 373a085 before any cross-stratum number existed.

  PYTHONPATH=. .venv/bin/python analyze_r2_sameness.py [results_dir]
  PYTHONPATH=. .venv/bin/python analyze_r2_sameness.py --selfcheck
  PYTHONPATH=. .venv/bin/python analyze_r2_sameness.py --controls   # failed runs only

The question. Gate 2 measured a clock inside each stratum, one stratum at a time. That
cannot distinguish "one mechanism, indexed per stratum" from "a private sub-circuit per
stratum", because both put a diagonal clock in every block. The discriminating question is
whether two strata that share a local group select the same characters of it.

The instrument already exists on one stratum: K_h = K_W at 14/14 (C31, analyze_o4.py),
two independently recovered key sets compared element for element. This lifts that
comparison between strata; nothing here is a new statistic, only a new pairing.

Why transposes are not support: (d,e) and (e,d) are transposes of one commutative task, so
their diagonal spectra agree for a reason that has nothing to do with a shared circuit.
They are scored and printed separately and never enter the primary rate.
"""
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

from analyze_gate2 import (D as GATE2_D, G_MIN, HELDOUT_MIN, TRAIN_FRAC, block, collapse,
                           discover, fibre_map, heldout_mask, jclass_dlog, path, run_state,
                           strata)
from src.analysis.sparsity import key_freqs_5x_median
from src import provenance

D = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else GATE2_D
CONTROLS = "--controls" in sys.argv
B_NULL = 10000          # R1's draw count, so every p in this project has one floor
RATE_MIN = 0.80         # R2-1
P_MAX = 0.01            # R2-2
SHUF_MAX = 0.30         # R2-3
CTRL_MAX = 0.50         # R2-4


def _neg(k, orders):
    return tuple((-x) % o for x, o in zip(k, orders))


def folded_labels(orders):
    """One canonical representative per conjugate pair of non-DC frequencies.

    A real block's spectrum obeys F[-k] = conj(F[k]), so k and -k are ONE frequency. The
    diagonal bin (k,k) has its conjugate at (-k,-k), which is the diagonal bin of -k.
    """
    zero, seen, out = (0,) * len(orders), set(), []
    for k in np.ndindex(*orders):
        if k == zero or k in seen:
            continue
        seen.add(k)
        seen.add(_neg(k, orders))
        out.append(k)
    return out


def diag_amplitude(C, orders):
    """Amplitude on the diagonal (kappa, kappa), DC dropped, conjugates folded.

    Amplitude, not energy: the Discrete-Log-Clock protocol this project reports every
    sparsity statistic under. Energy inflates every threshold-crossing
    statistic and would not be comparable to K_W, which is measured the same way.
    """
    r = len(orders)
    A = np.asarray(C).reshape(orders + orders + (-1,))
    F = np.fft.fftn(A, axes=tuple(range(2 * r)))
    E = (np.abs(F) ** 2).sum(-1)
    labels = folded_labels(orders)
    return labels, np.sqrt(np.array([E[k + k] for k in labels]))


def keyset(C, orders):
    """The stratum's key characters, by the project's existing detector.

    `key_freqs_5x_median` is 2606.17399's: above 5x the median. The same function, on the
    same amplitude convention, produces K_W in analyze_o4.py -- so a cross-stratum
    agreement here is measured by the instrument that produced the 14/14 it generalises.
    """
    labels, amp = diag_amplitude(C, orders)
    sel = key_freqs_5x_median(amp)
    return frozenset(labels[i] for i in sel), labels, amp


def block_keyset(L, n, d, e, m, shuffle_seed=None):
    rf, orders_m = fibre_map(n, d, m)
    cf, _ = fibre_map(n, e, m)
    g = int(np.prod(orders_m))
    Bk, _, _ = block(L, n, d, e)
    if shuffle_seed is not None:
        # R2-3: destroy the block's structure, keep every size and the marginal. The
        # control has to have the shape of the thing and not its content.
        rng = np.random.default_rng(shuffle_seed)
        sh = Bk.reshape(-1, Bk.shape[-1])
        Bk = sh[rng.permutation(len(sh))].reshape(Bk.shape)
    C, _ = collapse(Bk, rf, cf, g, g)
    return keyset(C, orders_m) + (orders_m,)


# --- the pairing --------------------------------------------------------------------

def pairs_for(n):
    """Unordered pairs of measurable strata sharing a local group, split by transposition.

    Structural: computed from the algebra alone, before any model is opened.
    """
    by = defaultdict(list)
    for s in strata(n):
        if s["measurable"]:
            by[(s["local"], tuple(s["orders_m"]))].append((s["d"], s["e"]))
    prim, trans = [], []
    for key, members in by.items():
        for a, b in itertools.combinations(sorted(members), 2):
            (trans if a == (b[1], b[0]) else prim).append((key, a, b))
    return prim, trans


def null_p(groups, observed, rng):
    """P(>= observed agreements | independent uniform key sets of the observed sizes).

    NOT a product of per-pair chance levels. Eight strata on one local group make 28 pairs
    that SHARE strata, so the pairs are dependent and a product null would be wrong in the
    flattering direction. Each draw re-draws every stratum's key set once and recounts, so
    the dependency is carried by construction. Phipson-Smyth, like every other p here.
    """
    hits = 0
    for _ in range(B_NULL):
        agree = 0
        for members, n_bins, sizes in groups:
            drawn = {s: frozenset(map(tuple, rng.choice(n_bins[s], size=sizes[s],
                                                        replace=False)))
                     for s in members}
            for a, b in itertools.combinations(sorted(members), 2):
                agree += drawn[a] == drawn[b]
        hits += agree >= observed
    return (1.0 + hits) / (B_NULL + 1.0)


def score_run(f, n, seed, shuffle=False):
    z = np.load(f, allow_pickle=False)
    if "logits_all" not in z.files:
        return None
    L = z["logits_all"]
    prim, trans = pairs_for(n)
    if not prim and not trans:
        return dict(n=n, seed=seed, primary=[], transpose=[], empty=True)

    want = {a for _, a, b in prim + trans} | {b for _, a, b in prim + trans}
    ks, labs = {}, {}
    for (d, e) in sorted(want):
        m = next(s["m"] for s in strata(n) if (s["d"], s["e"]) == (d, e))
        K, labels, _, orders = block_keyset(
            L, n, d, e, m, shuffle_seed=(hash((n, seed, d, e)) % 2**31) if shuffle else None)
        ks[(d, e)] = K
        labs[(d, e)] = labels

    def rows(pl):
        out = []
        for key, a, b in pl:
            ka, kb = ks[a], ks[b]
            j = (len(ka & kb) / len(ka | kb)) if (ka | kb) else float("nan")
            # Two empty key sets compare equal, and that is not agreement. White noise
            # has no bin above 5x its median, so an undetected clock would score as a
            # perfect match. R2-6 excludes these pairs and counts them.
            out.append(dict(local=key[0], orders=list(key[1]), a=list(a), b=list(b),
                            na=len(ka), nb=len(kb), equal=bool(ka == kb and ka), jaccard=j,
                            empty=not (ka and kb), bins=len(labs[a])))
        return out

    return dict(n=n, seed=seed, primary=rows(prim), transpose=rows(trans),
                keysets={f"{d}x{e}": sorted(map(list, K)) for (d, e), K in ks.items()})


# --- self-check: plant a frequency, never reason about a bin -------------------------

def _selfcheck():
    """Three plantings. If any fails, no number this script prints means anything."""
    for orders in [(20,), (2, 10), (4, 10)]:
        g = int(np.prod(orders))
        labels = folded_labels(orders)
        for k0 in (labels[0], labels[len(labels) // 2], labels[-1]):
            # a pure diagonal character: a function of the SUM of exponent coordinates
            ea = np.array(list(np.ndindex(*orders)))
            ph = 2 * np.pi * ((ea * np.array(k0)) / np.array(orders)).sum(1)
            pure = np.cos(ph[:, None] + ph[None, :])[..., None]

            # (a) The bin convention: plant a frequency and check where it lands. This is
            # independent of any threshold, so the detector cannot rescue or break it.
            labs, amp = diag_amplitude(pure, orders)
            assert labs[int(np.argmax(amp))] == tuple(k0), (orders, k0, labs[int(np.argmax(amp))])

            # (b) THE DETECTOR, on a spectrum shaped like a real one. A PERFECTLY pure
            # spectrum has median 0, and "above 5x the median" then selects every bin
            # holding float noise -- which is what this assertion caught on the Nyquist
            # bin at orders=(20,). Real blocks never have a zero median, so the planting
            # gets a broadband floor and the degenerate case is asserted separately below.
            rng = np.random.default_rng(7)
            noisy = pure + 0.02 * rng.standard_normal(pure.shape)
            K, _, _ = keyset(noisy, orders)
            assert K == {tuple(k0)}, (orders, k0, sorted(K))
        print(f"  selfcheck PASS  planted diagonal character lands in its own bin and is "
              f"recovered, orders={orders} |G|={g} folded bins={len(labels)}")

    # The degenerate case, asserted rather than avoided: a zero median makes the 5x rule
    # select everything above zero. It cannot arise on a real block (amplitudes summed over
    # n_out outputs are strictly positive), and every stratum's median is printed so the
    # claim is checkable rather than asserted.
    ea = np.array(list(np.ndindex(20)))
    ph = 2 * np.pi * ea[:, 0] * 10 / 20.0
    pure = np.cos(ph[:, None] + ph[None, :])[..., None]
    _, amp = diag_amplitude(pure, (20,))
    assert np.median(amp) == 0.0 and len(key_freqs_5x_median(amp)) > 1
    print("  selfcheck PASS  a zero-median spectrum is degenerate for the 5x rule, "
          "by construction and not by accident")

    # White noise has no key set, and two empty sets compare equal, so a pair of strata
    # with no detectable clock would score as a perfect match. Assert the emptiness, and
    # that the exclusion rule (R2-6) is what keeps it out of the rate.
    rng = np.random.default_rng(0)
    orders = (20,)
    empty = sum(not keyset(rng.standard_normal((20, 20, 4)), orders)[0] for _ in range(20))
    assert empty == 20, f"{20 - empty}/20 noise blocks produced a key set"
    print(f"  selfcheck PASS  white noise yields NO key set {empty}/20 times, so every "
          f"noise pair is excluded rather than counted as agreement")

    # The discriminating case: two blocks running different clocks must not agree. Without
    # this, a statistic that always says "equal" would pass every other check here.
    labels = folded_labels(orders)
    ea = np.array(list(np.ndindex(*orders)))
    def planted(k, seed):
        ph = 2 * np.pi * ((ea * np.array(k)) / np.array(orders)).sum(1)
        return (np.cos(ph[:, None] + ph[None, :])[..., None]
                + 0.02 * np.random.default_rng(seed).standard_normal((20, 20, 1)))
    same = keyset(planted(labels[2], 1), orders)[0] == keyset(planted(labels[2], 2), orders)[0]
    diff = keyset(planted(labels[2], 1), orders)[0] == keyset(planted(labels[5], 2), orders)[0]
    assert same and not diff, (same, diff)
    print("  selfcheck PASS  same planted clock -> EQUAL; different planted clock -> NOT")

    # the transpose split must be exactly what the algebra says
    # The algebra gives 28 + 6; the pre-registration's hand count said 30 + 4 (it missed
    # the transposes in G_33 and G_55; the total, 34 same-group pairs, was right). The
    # correction is in that file's Outcome section and is asserted here.
    prim, trans = pairs_for(165)
    assert (len(prim), len(trans)) == (28, 6), (len(prim), len(trans))
    assert (len(pairs_for(121)[0]), len(pairs_for(121)[1])) == (0, 1)
    assert (len(pairs_for(119)[0]), len(pairs_for(119)[1])) == (2, 1)
    assert (len(pairs_for(120)[0]), len(pairs_for(120)[1])) == (2, 2)
    assert (len(pairs_for(113)[0]), len(pairs_for(113)[1])) == (0, 0)
    print("  selfcheck PASS  structural pair counts: 165 -> 28+6, 120 -> 2+2, 119 -> 2+1, "
          "121 -> 0+1, 113 -> 0+0")
    print("analyze_r2_sameness selfcheck PASS")


def main():
    runs = []
    for n, seed in discover(D):
        f = path(D, n, seed)
        z = np.load(f, allow_pickle=False)
        state, acc = run_state(D, n, seed, z)
        want = "FAILED" if CONTROLS else "grokked"
        if state != want:
            print(f"  n={n} s{seed}: skip -- {state} (acc {acc:.4f}), this arm scores {want}")
            continue
        r = score_run(f, n, seed)
        if r is None:
            print(f"  n={n} s{seed}: skip -- logits_all NOT SAVED")
            continue
        r["state"], r["acc"] = state, acc
        r["shuffled"] = score_run(f, n, seed, shuffle=True)["primary"] if r["primary"] else []
        runs.append(r)
        p = r["primary"]
        print(f"  n={n} s{seed} {state} acc={acc:.4f}  primary pairs={len(p):3d} "
              f"equal={sum(x['equal'] for x in p):3d}  transpose={len(r['transpose'])} "
              f"equal={sum(x['equal'] for x in r['transpose'])}")

    prim_all = [x for r in runs for x in r["primary"]]
    prim = [x for x in prim_all if not x["empty"]]
    # The control keeps its empty sets in the denominator. `equal` is already false for an
    # empty set, so a shuffled block with no detectable clock counts as a non-agreement.
    # Dropping them would divide by zero and print `nan`, which is indistinguishable from a
    # control that failed.
    shuf = [x for r in runs for x in r["shuffled"]]
    trans = [x for r in runs for x in r["transpose"] if not x["empty"]]
    n_empty = len(prim_all) - len(prim)
    if not prim:
        print("\n  NO PRIMARY PAIRS -- R2 is NOT SCORABLE on this arm, and that is the result.")
        return runs, None

    n_eq = sum(x["equal"] for x in prim)
    rate = n_eq / len(prim)
    rate_sh = (sum(x["equal"] for x in shuf) / len(shuf)) if shuf else float("nan")
    assert shuf, "no shuffled pairs -- the control did not run, which is not the same as passing"

    groups = []
    for r in runs:
        by = defaultdict(set)
        for x in r["primary"]:
            by[(x["local"], tuple(x["orders"]))] |= {tuple(x["a"]), tuple(x["b"])}
        for key, members in by.items():
            sizes, bins = {}, {}
            for x in r["primary"]:
                for side in ("a", "b"):
                    s = tuple(x[side])
                    sizes[s] = x["na"] if side == "a" else x["nb"]
                    bins[s] = np.array(folded_labels(tuple(x["orders"])))
            groups.append((sorted(members), bins, sizes))
    p_null = null_p(groups, n_eq, np.random.default_rng(0))

    print(f"\n  R2-1 PRIMARY   exact key-set equality {n_eq}/{len(prim)} = {rate:.4f}"
          f"   threshold >= {RATE_MIN}   {'HELD' if rate >= RATE_MIN else 'NOT HELD'}")
    print(f"  R2-2 PRIMARY   Monte-Carlo null p = {p_null:.2e} (B={B_NULL}, floor "
          f"{1/(B_NULL+1):.1e})   threshold < {P_MAX}   "
          f"{'HELD' if p_null < P_MAX else 'NOT HELD'}")
    n_sh_keyed = sum(1 for x in shuf if not x["empty"])
    print(f"  R2-3 CONTROL   cell-shuffle agreement {sum(x['equal'] for x in shuf)}/{len(shuf)}"
          f" = {rate_sh:.4f}   must be < {SHUF_MAX}   "
          f"{'OK' if rate_sh < SHUF_MAX else 'CONTROL FAILED'}")
    print(f"                 ({n_sh_keyed} of {len(shuf)} shuffled blocks have ANY key set --"
          f" the shuffle removes the clock, it does not relabel it)")
    print(f"  R2-5 transpose pairs {sum(x['equal'] for x in trans)}/{len(trans)} equal"
          f"  -- DESCRIPTIVE ONLY, never support")
    med = int(np.median([x["na"] for x in prim] + [x["nb"] for x in prim]))
    print(f"  R2-6 median |K| = {med}; pairs EXCLUDED for an empty key set: "
          f"{n_empty} of {len(prim_all)}")
    print(f"       mean Jaccard on primary pairs = {np.nanmean([x['jaccard'] for x in prim]):.4f}")
    return runs, dict(n_pairs=len(prim), n_pairs_all=len(prim_all), n_empty=n_empty,
                      n_equal=n_eq, rate=rate, p_null=p_null,
                      rate_shuffled=rate_sh, n_transpose=len(trans),
                      n_transpose_equal=sum(x["equal"] for x in trans), median_K=med)


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        sys.exit(0)
    _selfcheck()
    print(f"\nR2 cross-stratum sameness -- {D} ({'FAILED controls' if CONTROLS else 'grokked'})")
    runs, summary = main()
    # R2_OUT_DIR exists because writing an untracked file into results/ while another
    # stamped sweep is still running sets git_dirty on that sweep's artifact.
    # Default is the repo; a scratch dir is for computing before the tree can take it.
    out = (os.environ.get("R2_OUT_DIR", "results/gate2").rstrip("/") + "/r2_sameness"
           + ("_controls" if CONTROLS else "") + "_"
           + D.rstrip("/").split("/")[-1] + ".json")
    # Stamp before opening the file. `open(out, "w")` creates the path; if it is new, it is
    # an untracked file, so `git status --porcelain` is non-empty and the artifact would
    # stamp git_dirty=True against a tree that was clean a moment earlier.
    prov = provenance.stamp(dict(B_NULL=B_NULL, RATE_MIN=RATE_MIN, P_MAX=P_MAX,
                                 SHUF_MAX=SHUF_MAX, results_dir=D, controls=CONTROLS))
    with open(out, "w") as fh:
        json.dump(dict(provenance=prov, runs=runs, summary=summary), fh, indent=1, default=str)
    assert provenance.read(out) is not None, "stamp must read back -- stamp() spreads its keys"

    print(f"\n  saved {out}")
