"""Post-grok transient excursions -- the instrument that retracts C27.

  PYTHONPATH=. .venv/bin/python analyze_excursions.py [results_dir]
  PYTHONPATH=. .venv/bin/python analyze_excursions.py --selfcheck

C27 read the final logged accuracy of k06's n=119 seed 0 (0.6798, down from 0.9989 two
hundred steps earlier) as a network that "de-grokked". It did not. Post-grok accuracy
spikes are common under wd=1.0: they occur in almost every run, last one logging
interval, and recover. The one that "did not recover" started on the last logged step,
where there was no next observation to recover into. That is right-censoring, not
de-grokking.

A statistic evaluated at one instant cannot distinguish a transient from a state, so this
asks the trajectory how often the transient happens and whether it ends, and marks an
excursion touching the final sample as censored.

hist columns are read by name: the kernels write test_acc at index 3, run_n7.py writes
train acc there.
"""
import glob, os, sys
import numpy as np

from src import provenance

D = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results/k06_horizon"
GROK, FLOOR = 0.99, 0.90      # grok gate (C3's), and the excursion floor


def hist_of(z):
    """step and test_acc, read by NAME. Returns (step, acc) or None."""
    if "hist" not in z.files or "hist_cols" not in z.files:
        return None
    h, cols = z["hist"], [str(c) for c in z["hist_cols"]]
    if "step" not in cols or "test_acc" not in cols:
        return None
    return h[:, cols.index("step")], h[:, cols.index("test_acc")]


def excursions(step, acc):
    """Post-grok dips below FLOOR. `censored` = the dip touches the last sample, so we
    never observed whether it recovers. That is the whole point of this script."""
    hit = np.flatnonzero(acc > GROK)
    if not len(hit):
        return None
    g = hit[0]
    post, ps = acc[g:], step[g:]
    bad, out, k = post < FLOOR, [], 0
    while k < len(bad):
        if not bad[k]:
            k += 1
            continue
        j = k
        while j < len(bad) and bad[j]:
            j += 1
        out.append(dict(start=int(ps[k]), samples=j - k, min_acc=float(post[k:j].min()),
                        censored=j >= len(bad)))
        k = j
    return dict(grok_step=int(ps[0]), n_post=len(post), ex=out)


def main():
    files = sorted(glob.glob(os.path.join(D, "WE_*.npz")))
    if not files:
        print(f"no WE_*.npz in {D}")
        return
    print("=" * 92)
    print(f"POST-GROK EXCURSIONS BELOW acc {FLOOR}   {D}")
    print("=" * 92)
    tot = rec = cen = nsamp = 0
    ungrokked = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        hv = hist_of(z)
        if hv is None:
            continue
        r = excursions(*hv)
        tag = os.path.basename(f)[3:-4]
        if r is None:
            ungrokked.append(tag)
            kind = "NEAR-GROK" if hv[1].max() > 0.90 else "FAILED"
            print(f"{tag:22s} {kind} (max acc {hv[1].max():.4f}) -- excluded. "
                  f"A near-grok is NOT a negative control (LAB_PROTOCOL.md).")
            continue
        nsamp += r["n_post"]
        marks = []
        for e in r["ex"]:
            tot += 1
            cen += e["censored"]
            rec += not e["censored"]
            marks.append(f"{e['start']}({e['samples']}s,{e['min_acc']:.3f})"
                         + ("*CENSORED*" if e["censored"] else ""))
        print(f"{tag:22s} grok@{r['grok_step']:6d}  {len(r['ex'])} excursion(s)"
              + ("  " + "  ".join(marks) if marks else ""))
    print("-" * 92)
    print(f"  {tot} excursions over {nsamp} post-grok samples "
          f"({100.0 * tot / max(nsamp, 1):.2f}% of samples)")
    print(f"  recovered within the run: {rec}      still down at the final sample: {cen}")
    if tot:
        longest = "1 logging interval" if cen + rec == tot else "see above"
        print(f"  every excursion listed above lasts {longest} unless marked otherwise")
    if ungrokked:
        print(f"  excluded (never grokked): {', '.join(ungrokked)}")
    print()
    print("READ THIS BEFORE QUOTING A FINAL-CHECKPOINT NUMBER FROM A CENSORED RUN:")
    print("  a *CENSORED* excursion means the saved final W_E/logits are a TRANSIENT,")
    print("  not the converged state. Any spectral number read off that artifact is")
    print("  measuring the spike. C27 did exactly that.")
    for f in files:
        z = np.load(f, allow_pickle=True)
        hv = hist_of(z)
        if hv is None:
            continue
        r = excursions(*hv)
        if r and any(e["censored"] for e in r["ex"]):
            print(f"  -> {f}")
            try:
                print(f"     provenance: {provenance.read(f)}")
            except Exception:
                pass


def _selfcheck():
    """The censoring logic is the claim, so it is what gets asserted."""
    step = np.arange(0, 2000, 200).astype(float)
    # grok at 200, one recovering dip at 800, ends healthy
    a = np.array([0.1, 0.995, 0.999, 0.999, 0.50, 0.999, 0.999, 0.999, 0.999, 0.999])
    r = excursions(step, a)
    assert len(r["ex"]) == 1, r
    assert r["ex"][0]["samples"] == 1 and not r["ex"][0]["censored"], r
    # same run, truncated one sample after the dip starts -> the SAME dip is censored
    r2 = excursions(step[:5], a[:5])
    assert len(r2["ex"]) == 1 and r2["ex"][0]["censored"], r2
    assert r2["ex"][0]["min_acc"] == 0.50
    # a run that never crosses the grok gate yields nothing rather than a false excursion
    assert excursions(step, np.full(10, 0.3)) is None
    # a two-sample dip is reported as two samples, not two excursions
    a3 = a.copy(); a3[5] = 0.40
    assert excursions(step, a3)["ex"][0]["samples"] == 2
    print("analyze_excursions selfcheck PASS -- censoring, recovery, gate and run-length")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
