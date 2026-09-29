"""N9 analysis -- scores the criteria pre-registered in
experiments/PREREGISTER_n9_precision.md. Committed BEFORE the data existed.

  PYTHONPATH=. .venv/bin/python analyze_n9.py [float32_results_dir]
  PYTHONPATH=. .venv/bin/python analyze_n9.py --selfcheck

Every statistic here is imported from analyze_n7, unmodified. That is the point: the
engine-vs-torch gap this experiment explains was first measured with these exact
functions, so re-implementing them would put a second variable into a one-variable A/B.
"""
import glob
import json
import os, sys
import numpy as np

import analyze_n7 as A
from src.analysis.sparsity import gini, participation_ratio, key_freqs_5x_median
from src import provenance

N = 113
SEEDS = [0, 1, 2]
D32 = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results/n9_f32"
D64 = "results/n7_engine"
TORCH = "results/k03_grid_acts"
TORCH_SEEDS = [0, 1, 2, 3, 4]
HELD, NOT_HELD = 0.70, 0.30        # the pre-registered fraction-explained thresholds


def arm_label(d, fallback):
    """Name the arm from the ARTIFACT'S OWN provenance, never from a hard-coded string.

    This file was written for N9, where D32 really was the engine at float32, so the label
    "engine float32" was baked in. O18 then pointed the same scorer at a PyTorch-on-CPU run
    (`analyze_n9.py results/o18_cpu`, which O18's pre-registration names as its analysis
    plan) and the output confidently called it "engine float32" -- a log that would tell a
    future session the exact opposite of what the run was. The npz already records
    `config.engine`; read it.
    """
    import glob as _g
    for f in sorted(_g.glob(os.path.join(d, f"WE_*n{N}_s*.npz"))):
        p = provenance.read(f) or {}
        try:
            eng = json.loads(p.get("config", "{}")).get("engine")
        except Exception:
            eng = None
        if eng:
            who = {"from-scratch": "engine", "pytorch-autograd": "torch "}.get(eng, eng)
            # The DIRECTORY has to be in the label too: O18 and k03 are both
            # "pytorch-autograd", and two rows reading "torch" are not two rows.
            return f"{who} {os.path.basename(d.rstrip('/')):<14}"
    return fallback


# Arm labels, read from each artifact's own provenance. N9's D32 was the engine at
# float32; O18's is PyTorch on CPU. The scorer must not assert which.
L32 = None  # set in __main__ once D32 is known
L64 = None  # set in __main__
LT = None   # set in __main__


def load32(s, d=None):
    return A.load(N, s, d or D32)


def load_torch(s):
    f = f"{TORCH}/WE_B_thesis_n{N}_s{s}.npz"
    return np.load(f, allow_pickle=False) if os.path.exists(f) else None


def stats(z):
    """The per-run row. Gini on amplitude, PR on energy; the two conventions are not
    interchangeable (amplitude reads PR 12.5 where energy reads 4.3)."""
    W = z["W_E"][:N].astype(float)
    fm = A.mult_amplitude(W, N)
    t = A.tail_gini(z, N)
    h, c = z["hist"], [str(x) for x in z["hist_cols"]]
    acc, st = h[:, c.index("test_acc")], h[:, c.index("step")]
    return dict(
        tail=None if t is None else t[0],
        tail_sd=None if t is None else t[1],
        tail_pts=None if t is None else t[2],
        gini_mult=gini(fm),
        gini_add=gini(A.add_amplitude(W, N)),
        gini_rand=gini(A.rand_amplitude(W, N)),
        pr=participation_ratio(fm ** 2),
        n_key=len(key_freqs_5x_median(fm)),
        wnorm=float(np.linalg.norm(W)),
        acc=float(acc[-1]),
        grok=int(st[np.argmax(acc > 0.99)]) if (acc > 0.99).any() else None,
        step=int(z["step"]) if "step" in z.files else int(st[-1]),
        grokked=A.grokked(z),
    )


# Denominator floors: 3x the standard error of the between-regime mean difference,
# computed from the spreads on disk before the run. Gini: per-run sd ~0.02 over 3 and 5
# seeds -> SEM ~0.014 -> floor 0.05, comfortably under the observed 0.187 gap. |W_E|:
# torch sd ~0.9, engine ~0.35 -> SEM ~0.45 -> floor 1.5, under the observed ~3.5 gap.
# A ratio is not a statistic when its denominator is free to approach the noise -- the
# C22/C23 control-mean defect in a different coat.
FLOOR_GINI, FLOOR_WNORM = 0.05, 1.5


def fraction_explained(m64, m32, mt, floor=FLOOR_GINI):
    """f = (float64 - float32) / (float64 - torch). 1 = precision explains the whole
    engine-vs-torch gap, 0 = none of it. None when the reference gap is under `floor`,
    i.e. when there was no gap worth explaining in the first place."""
    den = m64 - mt
    if abs(den) < floor:
        return None
    return (m64 - m32) / den


def verdict(f):
    if f is None:
        return "NOT ASSESSABLE (reference gap too small for a ratio)"
    return ("HELD" if f >= HELD else "NOT HELD" if f <= NOT_HELD else "PARTIAL") + f" (f = {f:.2f})"


def collect():
    r32 = {s: load32(s) for s in SEEDS}
    r64 = {s: A.load(N, s, D64) for s in SEEDS}
    rt = {s: load_torch(s) for s in TORCH_SEEDS}
    return ({s: stats(z) for s, z in r32.items() if z is not None},
            {s: stats(z) for s, z in r64.items() if z is not None},
            {s: stats(z) for s, z in rt.items() if z is not None})


def _mean(rows, key, grokked_only=True):
    v = [r[key] for r in rows.values() if (r["grokked"] or not grokked_only)
         and r[key] is not None]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def p0(s32):
    print("\n" + "=" * 78 + "\nP0 -- does the engine still grok in float32?\n" + "=" * 78)
    print(f"{'seed':>6} {'final test acc':>16} {'grok step':>11} {'last step':>11}")
    for s in sorted(s32):
        r = s32[s]
        print(f"{s:>6} {r['acc']:>16.4f} {str(r['grok']):>11} {r['step']:>11}")
    g = sum(r["grokked"] for r in s32.values())
    ok = "HELD" if g >= 2 else ("NOT ASSESSABLE (no runs)" if not s32 else "VOID -- comparison cannot be scored")
    print(f"\n  grokked {g}/{len(s32)}  ->  P0 {ok}")
    return g >= 2


def p1(s32, s64, st):
    print("\n" + "=" * 78)
    print("P1 (PRIMARY) -- does precision explain the Gini_mult gap?\n" + "=" * 78)
    print("tail Gini_mult = mean over the final 10k steps; grokked runs only.\n")
    rows = [(L32, s32), (L64, s64), (LT, st)]
    for name, d in rows:
        cells = "  ".join(f"s{s}:{d[s]['tail']:.4f}" if d[s]["tail"] is not None else f"s{s}:--"
                          for s in sorted(d) if d[s]["grokked"])
        m, k = _mean(d, "tail")
        print(f"  {name:>16}  {cells}   mean {m:.4f} (n={k})" if m is not None
              else f"  {name:>16}  {cells}   mean -- (n=0)")
    m32, m64, mt = _mean(s32, "tail")[0], _mean(s64, "tail")[0], _mean(st, "tail")[0]
    if None in (m32, m64, mt):
        print("\n  P1 NOT ASSESSABLE -- a regime has no grokked run with a tail window")
        return None
    f = fraction_explained(m64, m32, mt)
    print(f"\n  {L64.strip()} {m64:.4f}  ->  {L32.strip()} {m32:.4f}   "
          f"({LT.strip()} {mt:.4f})")
    # N9's f asks how much of the engine-vs-torch gap the NEW arm closes. O18's
    # pre-registration states the complementary fraction -- how much the substrate
    # EXPLAINS -- so print both, named, or the two runs cannot be compared without
    # someone silently flipping a sign.
    denom = (m64 - mt) if abs(m64 - mt) > 1e-12 else float("nan")
    print(f"  fraction of the gap the new arm CLOSES  (N9's f)   = {(m64-m32)/denom:.3f}")
    print(f"  fraction the new arm still CARRIES      (O18's f)  = {(m32-mt)/denom:.3f}")
    print(f"  P1 {verdict(f)}    [HELD >= {HELD}, NOT HELD <= {NOT_HELD}]")
    return f


def p2(s32, s64, st):
    print("\n" + "=" * 78)
    print("P2 -- does precision explain the embedding-norm gap?\n" + "=" * 78)
    m32, m64, mt = (_mean(d, "wnorm")[0] for d in (s32, s64, st))
    for name, m in ((L32, m32), (L64, m64), (LT, mt)):
        print(f"  {name:>16}  |W_E| mean {m:.3f}" if m is not None else f"  {name:>16}  --")
    if None in (m32, m64, mt):
        print("\n  P2 NOT ASSESSABLE")
        return None
    f = fraction_explained(m64, m32, mt, floor=FLOOR_WNORM)
    print(f"\n  P2 {verdict(f)}")
    return f


def p3(s32):
    print("\n" + "=" * 78)
    print("P3 -- is the float32 sparsity still MULTIPLICATIVE?\n" + "=" * 78)
    print("LAB_PROTOCOL.md: never report Gini_mult alone -- a purely additive signal scores"
          " 0.53-0.63\nin the multiplicative basis.\n")
    print(f"{'seed':>6} {'Gini_mult':>11} {'Gini_add':>10} {'Gini_rand':>10} "
          f"{'PR(energy)':>11} {'n_key':>6} {'ok':>5}")
    ok = True
    for s in sorted(s32):
        r = s32[s]
        if not r["grokked"]:
            print(f"{s:>6} {'-- not grokked, excluded (analyze_n7.grokked) --':>55}")
            continue
        good = r["gini_mult"] > r["gini_add"] and r["gini_mult"] > r["gini_rand"]
        ok &= good
        print(f"{s:>6} {r['gini_mult']:>11.4f} {r['gini_add']:>10.4f} {r['gini_rand']:>10.4f} "
              f"{r['pr']:>11.2f} {r['n_key']:>6} {'OK' if good else 'FAIL':>5}")
    print(f"\n  P3 {'HELD' if ok else 'NOT HELD'}")
    return ok


def exploratory(s32, s64, st):
    print("\n" + "=" * 78)
    print("EXPLORATORY -- no criterion attached, labelled as such in any write-up\n" + "=" * 78)
    print("The pre-registration makes NO prediction about grokking time: the two regimes"
          "\noverlap at n=113 on the data that existed before this run.\n")
    print(f"{'regime':>16} {'grok steps':>28} {'PR':>8} {'n_key':>7}")
    for name, d in ((L32, s32), (L64, s64), (LT, st)):
        g = "  ".join(str(d[s]["grok"]) for s in sorted(d) if d[s]["grokked"])
        pr, _ = _mean(d, "pr")
        nk, _ = _mean(d, "n_key")
        print(f"  {name:>14}  {g:>28} {pr:>8.2f} {nk:>7.2f}" if pr is not None
              else f"  {name:>14}  {g:>28}       --      --")


def provenance_block():
    print("\n" + "=" * 78 + "\nPROVENANCE\n" + "=" * 78)
    # Two namings in play: the engine writes WE_engine_*, the kernels and run_o18_cpu.py
    # write WE_B_thesis_*. The provenance block must find a run under EITHER, or it
    # silently reports a directory as empty.
    for label, d, pat in ((L32, D32, f"WE_*n{N}_s%d.npz"),
                          (L64, D64, f"WE_*n{N}_s%d.npz"),
                          (LT, TORCH, f"WE_B_thesis_n{N}_s%d.npz")):
        for s in (SEEDS if label != LT else TORCH_SEEDS):
            hits = sorted(glob.glob(os.path.join(d, pat % s)))
            if not hits:
                continue
            f = hits[0]
            p = provenance.read(f)
            sha = str(p.get("git_sha", "?"))[:7] if p else "?"
            dirty = p.get("git_dirty") if p else "?"
            cfg = p.get("config", "") if p else ""
            # Parse, never substring-match two spellings: torch stamps "torch.float64"
            # and the engine stamps "float64", and a spelling this did not know read as
            # "unstamped" -- indistinguishable from a run that recorded nothing.
            dt = json.loads(cfg).get("dtype", "unstamped").replace("torch.", "")
            print(f"  {label} s{s}  sha {sha}  dirty {dirty}  dtype {dt}")


def _selfcheck():
    """The arithmetic the verdict rests on, and the guard that stops a ratio being
    reported over a near-zero denominator."""
    assert abs(fraction_explained(0.75, 0.57, 0.57) - 1.0) < 1e-12
    assert abs(fraction_explained(0.75, 0.75, 0.57) - 0.0) < 1e-12
    assert abs(fraction_explained(0.75, 0.66, 0.57) - 0.5) < 1e-12
    assert fraction_explained(0.57, 0.57, 0.56) is None, "ratio over a near-zero gap"
    # the floor must not reject the gap this experiment exists to explain
    assert fraction_explained(0.7536, 0.60, 0.5665) is not None, "floor swallows the real gap"
    assert fraction_explained(11.6, 15.0, 15.1, floor=FLOOR_WNORM) is not None
    assert verdict(0.8).startswith("HELD") and verdict(0.2).startswith("NOT HELD")
    assert verdict(0.5).startswith("PARTIAL") and verdict(None).startswith("NOT ASSESSABLE")
    # the reference numbers quoted in the pre-registration, recomputed from disk
    s64 = {s: stats(z) for s in SEEDS if (z := A.load(N, s, D64)) is not None}
    st = {s: stats(z) for s in TORCH_SEEDS if (z := load_torch(s)) is not None}
    if s64 and st:
        m64, mt = _mean(s64, "tail")[0], _mean(st, "tail")[0]
        assert abs(m64 - 0.7536) < 5e-3, f"float64 reference moved: {m64:.4f}"
        assert abs(mt - 0.5665) < 5e-3, f"torch reference moved: {mt:.4f}"
        print(f"  reference bands reproduce: float64 {m64:.4f}, torch {mt:.4f}   OK")
    print("analyze_n9: SELFCHECK PASS")


def main():
    s32, s64, st = collect()
    print(f"N9 -- precision A/B at n={N}.  float32 {D32}  vs  float64 {D64}  vs  torch {TORCH}")
    print(f"runs found: float32 {len(s32)}/3, float64 {len(s64)}/3, torch {len(st)}/5")
    ok0 = p0(s32)
    if not ok0:
        print("\nP0 did not hold -- P1/P2 are VOID, not negative. An ungrokked run is the"
              "\nuntrained control, never a data point (analyze_n7.grokked).")
    f1 = p1(s32, s64, st) if ok0 else None
    f2 = p2(s32, s64, st) if ok0 else None
    ok3 = p3(s32) if ok0 else None
    exploratory(s32, s64, st)
    provenance_block()
    print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78)
    print(f"  P0 grok        {'HELD' if ok0 else 'NOT HELD -- rest is void'}")
    print(f"  P1 Gini_mult   {verdict(f1) if ok0 else 'VOID'}")
    print(f"  P2 |W_E|       {verdict(f2) if ok0 else 'VOID'}")
    print(f"  P3 mult basis  {('HELD' if ok3 else 'NOT HELD') if ok0 else 'VOID'}")
    print("\nFill the Outcome section of experiments/PREREGISTER_n9_precision.md, and do"
          "\nnot edit anything above it.")


if __name__ == "__main__":
    L32 = arm_label(D32, "arm-32")
    L64 = arm_label(D64, "arm-64")
    LT = arm_label(TORCH, "torch-ref")
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
