"""k01 SCOUT -- de-risking only, NOT paper data.

One question: does `a*b mod n` grok at all, for each n in the design? If composite
moduli never grok, the thesis design changes, and that is a ~1 GPU-hour question we
should not defer six weeks behind the from-scratch autograd engine.

The architecture below is hand-written (attention, MLP, embeddings -- no
nn.MultiheadAttention, no nn.Transformer). Only the autograd is PyTorch's. Every line
of it carries over verbatim once the from-scratch engine lands; this run gets redone
on that engine before anything is claimed.

Config matches arXiv 2607.07066 exactly so numbers are directly comparable:
1-layer decoder-only, d_model=128, d_mlp=512, ReLU, no layernorm, AdamW(lr=1e-3,
wd=1.0, betas=(0.9,0.98)), full batch, 30% train, 25k epochs.
"""
import json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "results/k01_scout"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available() or os.environ.get("ALLOW_CPU"), "no GPU"
if DEV == "cuda":
    cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
    assert cap in torch.cuda.get_arch_list(), f"{cap} unsupported by torch {torch.__version__}"

MODULI    = [121, 125, 113, 165, 119, 120]   # novel cell first: if cut, we keep what matters
STEPS     = 25_000
D_MODEL, N_HEADS, D_HEAD, D_MLP = 128, 4, 32, 512
TRAIN_FRAC, LR, WD, BETAS = 0.30, 1e-3, 1.0, (0.9, 0.98)
LOG_EVERY = 100
WALL_LIMIT = 5.0 * 3600      # self-imposed guard, well under the session cap
T0 = time.time()


class Transformer(torch.nn.Module):
    """1-layer decoder-only, hand-written. No layernorm, no biases (Nanda's setup)."""

    def __init__(self, n, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        def p(*shape, scale):
            return torch.nn.Parameter(torch.randn(*shape, generator=g) * scale)
        s = 1 / math.sqrt(D_MODEL)
        self.W_E = p(n + 1, D_MODEL, scale=s)          # tokens 0..n-1 plus "="
        self.W_pos = p(3, D_MODEL, scale=s)
        self.W_Q = p(N_HEADS, D_MODEL, D_HEAD, scale=s)
        self.W_K = p(N_HEADS, D_MODEL, D_HEAD, scale=s)
        self.W_V = p(N_HEADS, D_MODEL, D_HEAD, scale=s)
        self.W_O = p(N_HEADS, D_HEAD, D_MODEL, scale=1 / math.sqrt(N_HEADS * D_HEAD))
        self.W_in = p(D_MODEL, D_MLP, scale=s)
        self.W_out = p(D_MLP, D_MODEL, scale=1 / math.sqrt(D_MLP))
        self.W_U = p(D_MODEL, n, scale=s)

    def forward(self, toks, return_acts=False):
        B, T = toks.shape
        x = self.W_E[toks] + self.W_pos[:T]                        # (B,T,d)
        q = torch.einsum("btd,hdk->bhtk", x, self.W_Q)
        k = torch.einsum("btd,hdk->bhtk", x, self.W_K)
        v = torch.einsum("btd,hdk->bhtk", x, self.W_V)
        att = torch.einsum("bhqk,bhtk->bhqt", q, k) / math.sqrt(D_HEAD)
        mask = torch.triu(torch.ones(T, T, device=toks.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        z = torch.einsum("bhqt,bhtk->bhqk", att, v)
        x = x + torch.einsum("bhqk,hkd->bqd", z, self.W_O)
        pre = x @ self.W_in
        mlp = F.relu(pre) @ self.W_out
        x = x + mlp
        logits = x[:, -1] @ self.W_U                               # read out at "="
        if return_acts:
            return logits, F.relu(pre)[:, -1], att
        return logits


def dataset(n, seed):
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    a, b = a.ravel(), b.ravel()
    toks = np.stack([a, b, np.full_like(a, n)], 1)                 # "=" token is id n
    y = (a * b) % n
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    cut = int(TRAIN_FRAC * len(y))
    tr, te = perm[:cut], perm[cut:]
    T = lambda z: torch.as_tensor(z, device=DEV)
    return (T(toks[tr]).long(), T(y[tr]).long(), T(toks[te]).long(), T(y[te]).long())


def run(n, seed=0):
    xtr, ytr, xte, yte = dataset(n, seed)
    model = Transformer(n, seed).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=BETAS)
    hist, grok_step = [], None
    for step in range(STEPS + 1):
        model.train()
        logits = model(xtr)
        loss = F.cross_entropy(logits, ytr)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % LOG_EVERY == 0:
            model.eval()
            with torch.no_grad():
                lo_te = model(xte)
                te_loss = F.cross_entropy(lo_te, yte).item()
                te_acc = (lo_te.argmax(-1) == yte).float().mean().item()
                tr_acc = (logits.argmax(-1) == ytr).float().mean().item()
                maxlogit = logits.abs().max().item()
                wnorm = sum((p ** 2).sum() for p in model.parameters()).sqrt().item()
            hist.append(dict(step=step, train_loss=loss.item(), test_loss=te_loss,
                             train_acc=tr_acc, test_acc=te_acc,
                             max_logit=maxlogit, weight_norm=wnorm))
            if grok_step is None and te_acc > 0.99:
                grok_step = step
            if step % 2500 == 0:
                print(f"  n={n} step {step:6d} tr_acc={tr_acc:.3f} te_acc={te_acc:.3f} "
                      f"te_loss={te_loss:.4f} |w|={wnorm:.1f} maxlogit={maxlogit:.1f}",
                      flush=True)
        if not math.isfinite(loss.item()):
            print(f"  n={n} DIVERGED at step {step}", flush=True); break
        if time.time() - T0 > WALL_LIMIT:
            print(f"  n={n} wall guard hit at step {step}", flush=True); break

    np.savez_compressed(f"{OUT}/scout_n{n}_seed{seed}.npz",
                        W_E=model.W_E.detach().cpu().numpy(),
                        W_U=model.W_U.detach().cpu().numpy(),
                        hist=np.array([[h[k] for k in hist[0]] for h in hist]),
                        hist_cols=np.array(list(hist[0])))
    final = hist[-1]
    return dict(n=n, seed=seed, grok_step=grok_step, final_test_acc=final["test_acc"],
                final_train_acc=final["train_acc"], final_test_loss=final["test_loss"],
                max_logit=final["max_logit"], steps_run=final["step"],
                elapsed_s=round(time.time() - T0, 1))


results = []
for n in MODULI:
    if time.time() - T0 > WALL_LIMIT:
        print(f"wall guard: skipping n={n}", flush=True); continue
    print(f"=== n={n} ===", flush=True)
    r = run(n)
    results.append(r)
    print(f"  -> grok_step={r['grok_step']} final_test_acc={r['final_test_acc']:.4f}", flush=True)
    with open(f"{OUT}/scout_summary.json", "w") as f:      # incremental, per design rule 2
        json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
