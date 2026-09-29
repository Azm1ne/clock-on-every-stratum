#!/usr/bin/env python
"""Render every checkpoint we have. One folder of panels per run, plus a browsable index.

    PYTHONPATH=. .venv/bin/python scripts/render_all.py

Renders whatever each checkpoint supports and says plainly what it cannot render, so a
missing view is visibly a gap in what was SAVED rather than an absent phenomenon.
"""
import re, sys, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
from src.tasks.algebra import describe
from src.analysis.sparsity import gini, participation_ratio, key_freqs_5x_median
from src.analysis.transforms import unit_index
from src.viz import mechinterp as M
from src.viz import plots
from src.viz.plots import training_curves

FIG = Path("figures")


def modulus_of(path):
    m = re.search(r"[_n](\d{2,4})[_.]", Path(path).name)
    return int(m.group(1)) if m else None


def freq_norms(W, n):
    k = np.arange(1, n // 2 + 1)[:, None]; t = np.arange(n)[None, :]
    s = np.sin(2 * np.pi * k * t / n) @ W
    c = np.cos(2 * np.pi * k * t / n) @ W
    return np.sqrt((s ** 2).sum(1) + (c ** 2).sum(1))


def render(path, n, outdir, mul=True):
    outdir.mkdir(parents=True, exist_ok=True)
    z = np.load(path, allow_pickle=False)
    made, notes = [], []
    W = z["W_E"][:n].astype(float); W -= W.mean(0, keepdims=True)
    fn = freq_norms(W, n)
    idx, orders, _ = unit_index(n)
    U = np.array(sorted(idx, key=lambda x: idx[x]))
    # Protocol invariant 2 (O1, LAB_NOTEBOOK Entry 16): Gini goes on amplitude, the
    # participation ratio on normalised energy. `freq_norms` returns amplitude, so the PR
    # takes fn**2. Amplitude gives 12.5 where the correct value is 4.3, and the additive
    # column cannot tell the two conventions apart.
    row = dict(n=n, gini_add=gini(fn), pr_add=participation_ratio(fn ** 2),
               key_add=[k + 1 for k in key_freqs_5x_median(fn)])
    if len(orders) == 1:
        fm = freq_norms(W[U], len(U))
        row.update(gini_mult=gini(fm), pr_mult=participation_ratio(fm ** 2),
                   key_mult=[k + 1 for k in key_freqs_5x_median(fm)])
    else:
        row.update(gini_mult=None, pr_mult=None, key_mult=None)

    def attempt(fn_, *a, **kw):
        try:
            r = fn_(*a, **kw)
            if r:
                made.append(r)
            else:
                notes.append(f"{fn_.__name__}: not applicable")
        except Exception as e:
            notes.append(f"{fn_.__name__}: {type(e).__name__} {e}")

    attempt(M.embedding_structure, path, n, out=str(outdir / "embedding.png"))
    if "hist" in z.files:
        attempt(training_curves, path, out=str(outdir / "training.png"))
    if "mlp_acts" in z.files:
        attempt(M.neuron_heatmaps, path, n, out=str(outdir / "neurons.png"))
        attempt(M.neuron_heatmaps, path, n, dlog=True, out=str(outdir / "neurons_dlog.png"))
        # `a+b`/`a-b` term names are meaningless on a MULTIPLICATION run in raw residue
        # coordinates; in discrete-log coordinates the same model reads 100% a+b (C21).
        attempt(M.neuron_frequency_map, path, n, dlog=mul,
                out=str(outdir / "neuron_freqs.png"))
        # Returns None when there is no discrete-log coordinate (non-cyclic unit group).
        # Guarding it matters more than the missing column: without this the exception
        # propagates out of render() and the checkpoint loses EVERY panel, not just this
        # one -- which reads as "nothing was saved" rather than "one view is undefined".
        f = M.neuron_term_decomposition(path, n, verbose=False, dlog=mul)
        if f is None:
            notes.append("neuron_term_decomposition: non-cyclic unit group, "
                         "no discrete-log coordinate -- a+b/a-b are undefined here")
        else:
            row.update({f"term_{k}": float(v.mean()) for k, v in f.items()})
            row["composition"] = f["a+b"].mean() / max(f["a+b"].mean() + f["a-b"].mean(), 1e-12)
    else:
        notes.append("mlp_acts NOT SAVED -- no neuron views")
    if "logits_all" in z.files:
        attempt(M.logit_spectrum, path, n, out=str(outdir / "logits.png"))
        top = M.logit_top_components(path, n, top=8, verbose=False)
        row["logit_top"] = [(a, b, k) for a, b, _, k in top[:4]]
    else:
        notes.append("logits_all NOT SAVED -- no logit views")
    if "attn" not in z.files:
        notes.append("attn NOT SAVED -- no attention views")
    return made, notes, row


# The paper's figures: (name, plots.py function, kwargs, probe path, description).
# One list, so the renderer and any check on the figures agree on which figures exist,
# and each figure carries its own description.
PAPER_FIGURES = [
    ("gate2_strata", "gate2_strata", {}, "results/gate2/k03_grid_acts.json",
     "every algebraic stratum against its own control (C33, Gate 2)"),
    # omega_bands reads every results/k0*/WE_*.npz via analyze_omega.collect, so
    # its "source" is the results tree itself; k02 stands in as the probe.
    ("omega_bands", "omega_bands", {}, "results/k02_grid",
     "additive sparsity by omega(n) vs the zero-divisor-density confound, with "
     "the random-orthogonal size control (PREREGISTER_omega.md)"),
    # Every figure the paper uses regenerates here; a figure function with no caller
    # cannot be rebuilt from a fresh clone.
    ("F0_pipeline", "pipeline", {}, "src/tasks/algebra.py",
     "what the paper does end to end; the one non-standard step is the "
     "exponent coordinate, taken per stratum"),
    ("F1_stratum", "stratum_theorem", {}, "src/tasks/algebra.py",
     "Theorem 1 and the limitation it answers: a stratum unfolded into "
     "multiplication on its local group, with one cell carried through"),
    ("F2_stratification", "stratification", {}, "src/tasks/algebra.py",
     "the same operation over five algebras; only the stratification differs "
     "(C33). Red outlines mark non-regular classes"),
    ("F3_ablation_scheme", "ablation_scheme", {},
     "results/k03_grid_acts/WE_B_thesis_n121_s0.npz",
     "what the causal test does to the character plane: baseline, restricted "
     "and excluded as three projections Pi_S (C22/C23)"),
    ("F4_causal", "causal_test", {}, "results/k03_grid_acts_n4_ablation.npz",
     "the causal test and its permutation null (C22/C23/C36)"),
    ("F5_descent", "descent_lattice", {}, "src/tasks/algebra.py",
     "how strata nest: squaring walks a class down the divisor lattice, and "
     "nu(n) is how far"),
    ("F7_precision", "precision_2x2", {}, "results/o18_f64",
     "the O18 2x2: precision is inert in the engine and decisive in torch "
     "(C34), every cell recomputed rather than quoted"),
    ("F6_crt", "crt_law", {}, "results/k01_scout",
     "C8: additive energy on the CRT-dual frequency set, against a 20,000-"
     "permutation null, with their published checkpoint as calibration"),
    ("F8_grok", "grok_curves", {}, "results/k03_grid_acts",
     "test accuracy per run with the three-state grokked/near/failed rule"),
    # basis_comparison, the thirteenth paper figure.
    ("basis_comparison",
     "basis_comparison", {"npz_path": "results/k03_grid_acts/WE_B_thesis_n113_s0.npz", "n": 113},
     "results/k03_grid_acts/WE_B_thesis_n113_s0.npz",
     "the same model in the multiplicative and additive bases: sparsity is a "
     "property of the coordinate, not of the model (C4)"),
    ("F9_discrimination", "gate2_discrimination", {}, "results/gate2",
     "which Gate 2 criteria separate a grokked model from a failed one -- "
     "including G1, the pre-registered primary, which does not"),
    ("F10_internal", "internal_intervention", {},
     "results/i1_internal/I1_engine_n113_s0.npz",
     "the intervention on W_E with the forward pass RE-RUN (C39), and the "
     "key-character dose-response that makes the null's heavy tail benign")]


def main():
    all_npz = sorted(Path("results").rglob("*.npz"))
    # `ckpt_*.npz` are optimiser resume state (Adam m_*/v_*, p_*, step), not results:
    # no W_E, no hist_cols, no provenance. They are excluded by name, and counted so the
    # exclusion is visible.
    ckpts = [c for c in all_npz if not c.name.startswith("ckpt_")]
    skipped_state = len(all_npz) - len(ckpts)
    rows = []
    print(f"{len(ckpts)} checkpoints"
          + (f"   ({skipped_state} ckpt_* resume-state files excluded -- not results)"
             if skipped_state else "") + "\n")
    for c in ckpts:
        n = modulus_of(c)
        if n is None:
            print(f"  {c}: cannot infer modulus, skipped"); continue
        out = FIG / c.parent.name / c.stem
        # gate1 is `a+b mod 113` -- raw residue coordinates are the right ones there.
        # Everything else in results/ is `a*b mod n` and needs the dlog relabelling.
        mul = c.parent.name != "gate1"
        try:
            made, notes, row = render(c, n, out, mul=mul)
        except Exception:
            print(f"  {c}: FAILED\n{traceback.format_exc(limit=1)}"); continue
        row["run"] = f"{c.parent.name}/{c.stem}"
        rows.append(row)
        print(f"  {row['run']:<38} n={n:<4} {len(made)} panels -> {out}/"
              + (f"   [{'; '.join(notes)}]" if notes else ""))

    # Cross-run panels: these read results/, not a single checkpoint, so they sit
    # outside the per-run loop. NOT SAVED is never absence -- say which one is missing.
    cross = []
    for name, fn, kw, src, desc in PAPER_FIGURES:
        if not Path(src).exists():
            print(f"  {name:<38} NOT SAVED -- {src} absent (run analyze_gate2.py)")
            continue
        try:
            cross.append((name, getattr(plots, fn)(**kw), desc))
            print(f"  {name:<38} -> {cross[-1][1]}")
        except Exception:
            print(f"  {name}: FAILED\n{traceback.format_exc(limit=1)}")

    # index: numbers beside the pictures, so a pattern in one can be checked in the other
    lines = ["# Figure index\n",
             "Regenerate: `PYTHONPATH=. .venv/bin/python scripts/render_all.py`\n",
             "| run | n | Gini add | Gini mult | key mult | a+b | a-b | comp | panels |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["n"], r["run"])):
        g = lambda k, f="{:.3f}": (f.format(r[k]) if r.get(k) is not None else "—")
        lines.append(
            f"| `{r['run']}` | {r['n']} | {g('gini_add')} | {g('gini_mult')} | "
            f"{r.get('key_mult') if r.get('key_mult') is not None else '—'} | "
            f"{g('term_a+b','{:.1%}')} | {g('term_a-b','{:.1%}')} | "
            f"{g('composition','{:.0%}')} | "
            f"[dir](./{Path(r['run']).parent}/{Path(r['run']).name}/) |")
    if cross:
        lines += ["", "## Cross-run panels", ""]
        lines += [f"- [`{n}`](./{Path(f).name}) — {d}" for n, f, d in cross]
    (FIG / "INDEX.md").write_text("\n".join(lines) + "\n")
    print(f"\nindex -> {FIG}/INDEX.md   ({len(rows)} runs)")


if __name__ == "__main__":
    main()
