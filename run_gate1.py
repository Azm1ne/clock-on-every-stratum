"""GATE 1 run: a+b mod 113 on the from-scratch engine. Zero Kaggle quota.

Success is defined by test_gate1.py, written before this run. Checkpoints every 500 steps
so the gate can be evaluated on partial results if the run is cut.
"""
import json, os, time
import numpy as np
from src.model.transformer import Transformer
from src.train.loop import modular_data
from src.autograd.engine import no_grad
from src.autograd.nn import cross_entropy, AdamW
from src.provenance import stamp_npz

N, STEPS, LOG, CKPT = 113, 25_000, 100, 2_500
OUT = "results/gate1"
os.makedirs(OUT, exist_ok=True)

xtr, ytr, xte, yte = modular_data(N, "add", train_frac=0.30, seed=0)
a, b = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
x_all = np.stack([a.ravel(), b.ravel(), np.full(N * N, N)], 1)
y_all = (a.ravel() + b.ravel()) % N

m = Transformer(N + 1, N, d_model=128, n_heads=4, d_head=32, d_mlp=512, seed=0)
opt = AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0)
print(f"n={N} add | {len(ytr)} train / {len(yte)} test | "
      f"{sum(p.data.size for p in m.parameters()):,} params", flush=True)

hist, t0, grok = [], time.time(), None


def snapshot(step, tag):
    # Chunked and under no_grad: the full grid is 3.3x the training batch and needs no
    # graph. Building one here is what took RSS to 10 GB against 14 GB of RAM.
    lgs, pres, atts = [], [], []
    with no_grad():
        for i in range(0, len(x_all), 2000):
            l, p_, at = m(x_all[i:i + 2000], return_acts=True)
            lgs.append(l.data.astype(np.float32))
            pres.append(np.maximum(p_.data, 0).astype(np.float32))
            atts.append(at.data[:, :, -1, :].astype(np.float32))   # from "=" to (a,b,=)
    lg_d, pre_d, att_d = (np.concatenate(x) for x in (lgs, pres, atts))
    cfg = dict(task="add", n=N, steps=STEPS, train_frac=0.30, d_model=128, n_heads=4,
               d_head=32, d_mlp=512, lr=1e-3, wd=1.0, betas=(0.9, 0.98), seed=0,
               engine="from-scratch", arm="gate1", at_step=step)
    np.savez_compressed(f"{OUT}/add113_{tag}.npz", **stamp_npz(cfg),
                        W_E=m.W_E.data, W_U=m.W_U.data,
                        logits_all=lg_d, mlp_acts=pre_d, attn=att_d,
                        y_all=y_all, step=step,
                        hist=np.array([[h[k] for k in hist[0]] for h in hist]),
                        hist_cols=np.array(list(hist[0])))


for step in range(STEPS + 1):
    opt.zero_grad()
    logits = m(xtr)
    loss = cross_entropy(logits, ytr)
    loss.backward()
    opt.step()

    if step % LOG == 0:
        with no_grad():
            lte = m(xte)
            te_acc = float((lte.data.argmax(-1) == yte).mean())
            te_loss = float(cross_entropy(lte, yte).data)
        rec = dict(step=step, train_loss=float(loss.data),
                   test_loss=te_loss,
                   train_acc=float((logits.data.argmax(-1) == ytr).mean()),
                   test_acc=te_acc, max_logit=float(np.abs(logits.data).max()),
                   weight_norm=float(np.sqrt(sum((p.data ** 2).sum()
                                                 for p in m.parameters()))))
        hist.append(rec)
        if grok is None and te_acc > 0.99:
            grok = step
            print(f"  *** GROKKED at step {step} ***", flush=True)
            snapshot(step, "atgrok")
        if step % 500 == 0:
            el = time.time() - t0
            print(f"  step {step:6d} tr {rec['train_acc']:.3f} te {te_acc:.3f} "
                  f"trloss {rec['train_loss']:.2e} teloss {rec['test_loss']:.3f} "
                  f"|w| {rec['weight_norm']:.1f} maxlgt {rec['max_logit']:.0f} "
                  f"[{el/60:.0f}m, eta {el/max(step,1)*(STEPS-step)/3600:.1f}h]", flush=True)
            with open(f"{OUT}/history.json", "w") as f:
                json.dump(dict(hist=hist, grok_step=grok), f)
    if step % CKPT == 0 and step > 0:
        snapshot(step, "final")          # rolling; the gate can run on it any time

snapshot(STEPS, "final")
print(f"done. grok_step={grok}, {(time.time()-t0)/3600:.2f} h", flush=True)
