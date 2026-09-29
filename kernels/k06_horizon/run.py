"""k06 (N10) -- the convergence horizon at the four moduli C24 flagged.

Criteria are fixed in experiments/PREREGISTER_k06_horizon.md, committed before this ran.

k03's H4 found the participation ratio still moving 4.8-7.7% over the FINAL 10k steps of a
40k run at 125, 119, 120 and 165 (C24). Every PR number at those moduli is therefore taken
at a non-stationary point. This is the only run in the project whose object is the
MEASUREMENT INSTRUMENT rather than the phenomenon: 120k epochs, 3x the k03 horizon, same
seeds, W_E snapshotted every 4k steps so drift can be computed at ANY horizon afterwards.

The step-40k slice pools with k03 (identical config and seeds). NOTHING past 40k pools with
anything -- it is a new horizon and must be labelled so wherever it is reported.

Provenance: __GIT_SHA__ and __KAGGLE_OWNER__ are substituted by run_kernel.py at push
time. Kaggle has no git and no access to our env, so an env-var lookup silently stamps
"unknown"; and with two accounts in play the artifact must say WHOSE quota ran it.
"""
import json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "results/k06_horizon"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "no GPU"
cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
assert cap in torch.cuda.get_arch_list(), f"{cap} unsupported by torch {torch.__version__}"
print(f"{torch.cuda.get_device_name(0)} {cap}", flush=True)

GIT_SHA = "__GIT_SHA__"                       # substituted by run_kernel.py
ACCOUNT = "__KAGGLE_OWNER__"                  # which Kaggle account ran this
STEPS = 120_000                              # 3x the k03 horizon; C24
SNAP_EVERY = 4_000                           # W_E trajectory for the convergence audit
MODULI = [125, 119, 120, 165]                 # exactly the four C24 flagged
SEEDS = [0, 1, 2]
D_MODEL, N_HEADS, D_HEAD, D_MLP = 128, 4, 32, 512
TRAIN_FRAC, LR, WD, BETAS = 0.30, 1e-3, 1.0, (0.9, 0.98)
WALL = 10.0 * 3600                           # 12 runs x 120k steps
T0 = time.time()
RESULTS = f"{OUT}/horizon_summary.json"
done = []
if os.path.exists(RESULTS):                  # resumable across pushes
    done = json.load(open(RESULTS))
seen = {(r["arm"], r["n"], r["seed"]) for r in done}


class Transformer(torch.nn.Module):
    """Same architecture as k01_scout and src/model/transformer.py."""

    def __init__(self, n_vocab, n_out, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        s = 1 / math.sqrt(D_MODEL)
        p = lambda *sh, sc=s: torch.nn.Parameter(torch.randn(*sh, generator=g) * sc)
        self.W_E, self.W_pos = p(n_vocab, D_MODEL), p(3, D_MODEL)
        self.W_Q, self.W_K, self.W_V = (p(N_HEADS, D_MODEL, D_HEAD) for _ in range(3))
        self.W_O = p(N_HEADS, D_HEAD, D_MODEL, sc=1 / math.sqrt(N_HEADS * D_HEAD))
        self.W_in = p(D_MODEL, D_MLP)
        self.W_out = p(D_MLP, D_MODEL, sc=1 / math.sqrt(D_MLP))
        self.W_U = p(D_MODEL, n_out)

    def forward(self, toks, acts=False):
        B, T = toks.shape
        x = self.W_E[toks] + self.W_pos[:T]
        q, k, v = (torch.einsum("btd,hdk->bhtk", x, W)
                   for W in (self.W_Q, self.W_K, self.W_V))
        a = torch.einsum("bhqk,bhtk->bhqt", q, k) / math.sqrt(D_HEAD)
        a = a.masked_fill(torch.triu(torch.ones(T, T, device=toks.device, dtype=torch.bool), 1),
                          float("-inf")).softmax(-1)
        z = torch.einsum("bhqt,bhtk->bhqk", a, v)
        x = x + torch.einsum("bhqk,hkd->bqd", z, self.W_O)
        last = x[:, -1]
        pre = last @ self.W_in
        logits = (last + F.relu(pre) @ self.W_out) @ self.W_U
        return (logits, F.relu(pre), a) if acts else logits


def dataset(n, seed, units_only):
    from math import gcd
    if units_only:
        vals = np.array([x for x in range(n) if gcd(x, n) == 1])
    else:
        vals = np.arange(n)
    a, b = np.meshgrid(vals, vals, indexing="ij")
    a, b = a.ravel(), b.ravel()
    toks = np.stack([a, b, np.full_like(a, n)], 1)
    y = (a * b) % n
    perm = np.random.default_rng(seed).permutation(len(y))
    cut = int(TRAIN_FRAC * len(y))
    T = lambda z: torch.as_tensor(z, device=DEV).long()
    return (T(toks[perm[:cut]]), T(y[perm[:cut]]),
            T(toks[perm[cut:]]), T(y[perm[cut:]]), toks, y)


def run(arm, n, seed, units_only, save_acts):
    xtr, ytr, xte, yte, toks_all, y_all = dataset(n, seed, units_only)
    m = Transformer(n + 1, n, seed).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD, betas=BETAS)
    grok, hist, snaps, snap_steps = None, [], [], []
    for step in range(STEPS + 1):
        logits = m(xtr)
        loss = F.cross_entropy(logits, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                lo = m(xte)
                te = (lo.argmax(-1) == yte).float().mean().item()
            hist.append((step, loss.item(), F.cross_entropy(lo, yte).item(), te))
            if grok is None and te > 0.99:
                grok = step
        if step % SNAP_EVERY == 0:
            snaps.append(m.W_E.detach().cpu().numpy().astype(np.float32))
            snap_steps.append(step)
        if step % 10000 == 0:
            print(f"    {arm} n={n} s={seed} step {step} te={hist[-1][3]:.3f} "
                  f"[{(time.time()-T0)/60:.0f}m]", flush=True)
        if time.time() - T0 > WALL:
            print("    wall guard", flush=True); break
    with torch.no_grad():
        lo, pre, att = m(torch.as_tensor(toks_all, device=DEV).long(), acts=True)
        acc = (lo.argmax(-1).cpu().numpy() == y_all).mean()
    tag = f"{arm}_n{n}_s{seed}"
    prov = {"provenance": np.array(json.dumps(dict(
        git_sha=GIT_SHA,
        kaggle_account=ACCOUNT,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config=json.dumps(dict(arm=arm, n=n, seed=seed, units_only=units_only,
                               steps=STEPS, train_frac=TRAIN_FRAC, lr=LR, wd=WD,
                               d_model=D_MODEL, engine="pytorch-autograd")))))}
    np.savez_compressed(f"{OUT}/WE_{tag}.npz", **prov, W_E=m.W_E.detach().cpu().numpy(),
                        W_U=m.W_U.detach().cpu().numpy(),
                        hist=np.array(hist),
                        hist_cols=np.array(["step", "train_loss", "test_loss", "test_acc"]),
                        we_traj=np.stack(snaps), we_traj_steps=np.array(snap_steps),
                        y_all=y_all,
                        **({"mlp_acts": pre.cpu().numpy().astype(np.float32),
                            "attn": att[:, :, -1, :].cpu().numpy().astype(np.float32),
                            "logits_all": lo.cpu().numpy().astype(np.float32)}
                           if save_acts else {}))
    return dict(arm=arm, n=n, seed=seed, units_only=units_only, steps=STEPS,
                grok_step=grok, final_acc_all_pairs=float(acc),
                final_test_acc=hist[-1][3], elapsed_s=round(time.time() - T0, 1))


jobs = []
jobs += [("B_thesis", n, s, False, True) for n in MODULI for s in SEEDS]   # acts every seed

for arm, n, seed, uo, sa in jobs:
    if (arm, n, seed) in seen:
        continue
    if time.time() - T0 > WALL:
        print("wall guard: stopping", flush=True); break
    print(f"=== {arm} n={n} seed={seed} units_only={uo} ===", flush=True)
    r = run(arm, n, seed, uo, sa)
    done.append(r)
    print(f"  -> grok {r['grok_step']} acc {r['final_test_acc']:.4f}", flush=True)
    with open(RESULTS, "w") as f:
        json.dump(done, f, indent=2)

print(json.dumps(done[-6:], indent=2))
