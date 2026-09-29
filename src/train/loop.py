"""Full-batch training loop on the from-scratch engine."""
import numpy as np
from ..autograd.nn import cross_entropy, AdamW


def modular_data(n, op="mul", train_frac=0.3, seed=0):
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    a, b = a.ravel(), b.ravel()
    toks = np.stack([a, b, np.full_like(a, n)], 1)
    y = (a * b if op == "mul" else a + b) % n
    perm = np.random.default_rng(seed).permutation(len(y))
    cut = int(train_frac * len(y))
    return toks[perm[:cut]], y[perm[:cut]], toks[perm[cut:]], y[perm[cut:]]


def train(model, xtr, ytr, xte, yte, steps=5000, lr=1e-3, wd=1.0, log_every=250,
          verbose=True, time_limit=None):
    import time
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    hist, t0 = [], time.time()
    for step in range(steps + 1):
        opt.zero_grad()
        logits = model(xtr)
        loss = cross_entropy(logits, ytr)
        loss.backward()
        opt.step()
        if step % log_every == 0:
            te = model(xte)
            rec = dict(step=step, train_loss=float(loss.data),
                       train_acc=float((logits.data.argmax(-1) == ytr).mean()),
                       test_acc=float((te.data.argmax(-1) == yte).mean()),
                       max_logit=float(np.abs(logits.data).max()))
            hist.append(rec)
            if verbose:
                print(f"  step {step:5d}  loss {rec['train_loss']:.4f}  "
                      f"tr {rec['train_acc']:.3f}  te {rec['test_acc']:.3f}  "
                      f"maxlogit {rec['max_logit']:.1f}  [{time.time()-t0:.0f}s]", flush=True)
        if time_limit and time.time() - t0 > time_limit:
            print(f"  time limit at step {step}"); break
    return hist
