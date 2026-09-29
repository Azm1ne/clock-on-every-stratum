"""C12/C13 -- Softmax Collapse on the from-scratch engine, with a saved artifact.

  PYTHONPATH=. .venv/bin/python run_c12_collapse.py

C12 was the only claim in the ledger with no artifact and no script: its evidence was a
printed table in FINDINGS.md 8.1 from an ad-hoc n=17 run that no longer exists. This is
that run, made reproducible -- same task (n=17, mul, train_frac 0.40), same engine, a
fixed seed, and a provenance-stamped npz.

Two arms, so C13 (StableMax mitigates) gets the same treatment:
  softmax    -- cross_entropy through softmax; the collapse arm
  stablemax  -- stablemax_cross_entropy (prieto2025grokking)

What is measured at each log step: train loss, GLOBAL gradient norm over all parameters,
max |logit|, and mean p_correct under the arm's own normaliser. Softmax Collapse is
|grad| falling by orders of magnitude while p_correct saturates at exactly 1.0 -- at which
point only weight decay still moves the weights.

NOTE: n=17 at train_frac 0.40 is BELOW critical dataset size (C14) and neither arm is
expected to generalize. That is not a StableMax failure and is not read as one here; this
script measures the OPTIMIZATION pathology, not generalization.
"""
import numpy as np

from src import provenance
from src.autograd import nn as N
from src.autograd.engine import no_grad
from src.model.transformer import Transformer
from src.train.loop import modular_data

N_MOD, TRAIN_FRAC, STEPS, LOG, SEED = 17, 0.40, 4000, 100, 0
LR, WD = 1e-3, 1.0
OUT = "results/c12_collapse/softmax_collapse_n17.npz"
COLS = ["step", "train_loss", "grad_norm", "max_logit", "p_correct", "train_acc", "test_acc"]


def run(arm):
    lossfn = N.cross_entropy if arm == "softmax" else N.stablemax_cross_entropy
    norm = N.softmax if arm == "softmax" else N.stablemax
    xtr, ytr, xte, yte = modular_data(N_MOD, "mul", TRAIN_FRAC, seed=SEED)
    m = Transformer(N_MOD + 1, N_MOD, seed=SEED)
    opt = N.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    hist = []
    for step in range(STEPS + 1):
        opt.zero_grad()
        logits = m(xtr)
        loss = lossfn(logits, ytr)
        loss.backward()
        if step % LOG == 0:
            g = float(np.sqrt(sum(float((p.grad ** 2).sum()) for p in m.parameters())))
            p = norm(logits).data if arm == "softmax" else norm(logits).data
            pc = float(p[np.arange(len(ytr)), ytr].mean())
            with no_grad():
                te = m(xte)
            hist.append([step, float(loss.data), g, float(np.abs(logits.data).max()), pc,
                         float((logits.data.argmax(-1) == ytr).mean()),
                         float((te.data.argmax(-1) == yte).mean())])
            print(f"  {arm:9s} {step:5d}  loss {hist[-1][1]:.3e}  |grad| {g:.3e}  "
                  f"maxlogit {hist[-1][3]:6.1f}  p_correct {pc:.6f}", flush=True)
        opt.step()
    return np.array(hist, float)


def main():
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    arms = {a: run(a) for a in ("softmax", "stablemax")}
    cfg = dict(n=N_MOD, op="mul", train_frac=TRAIN_FRAC, steps=STEPS, log_every=LOG,
               seed=SEED, lr=LR, wd=WD, engine="from-scratch",
               dtype=os.environ.get("ENGINE_DTYPE", "float64"))
    np.savez(OUT, hist_cols=np.array(COLS), softmax=arms["softmax"],
             stablemax=arms["stablemax"], **provenance.stamp_npz(cfg))
    assert provenance.read(OUT), "artifact must be readable by provenance.read"
    s, t = arms["softmax"], arms["stablemax"]
    gi = COLS.index("grad_norm"); pi = COLS.index("p_correct")
    print(f"\nwrote {OUT}")
    print(f"C12  softmax   |grad| {s[0, gi]:.3e} -> {s[-1, gi]:.3e}  "
          f"({np.log10(s[0, gi] / s[-1, gi]):.1f} orders), p_correct -> {s[-1, pi]:.6f}")
    print(f"C13  stablemax |grad| {t[0, gi]:.3e} -> {t[-1, gi]:.3e}  "
          f"({np.log10(t[0, gi] / t[-1, gi]):.1f} orders), p_correct -> {t[-1, pi]:.6f}")
    print(f"C13  ratio at step {STEPS}: stablemax/softmax |grad| = {t[-1, gi] / s[-1, gi]:.1f}x")


if __name__ == "__main__":
    main()
