"""k09 -- three primes and a zero-divisor-density axis. Two gaps, one push.

Criteria are fixed in experiments/PREREGISTER_k09_primes_zdd.md, committed before this ran.

GAP 1 (G1): THIS PROJECT HAS TRAINED EXACTLY ONE PRIME. 18 moduli, and `isprime` is true
for n=113 alone, so every "primes differ from composites" sentence rests on one run set
against a MEASURED grokking-time seed variance of 3.8x (O2). 127 and 131 make it three.

GAP 2 (W1): PREREGISTER_omega.md's PRIMARY criterion came back NOT ASSESSABLE -- only 5
modulus pairs are matched on zero-divisor density while differing in omega(n), against a
pre-registered minimum of 6. The threshold was not moved. Closing it needs MORE MODULI, not
a wider tolerance, so 128, 123 and 91 are chosen to sit beside existing moduli on the zdd
axis at contrasting omega. Computed before the run (analyze_omega.confirm): 5 pairs -> 10.

  n=128 = 2^7   omega 1, zdd 0.500  -> pairs with 75 (.033), 165 (.015), 105 (.043)
  n=123 = 3*41  omega 2, zdd 0.350  -> pairs with 81 (.016)
  n=91  = 7*13  omega 2, zdd 0.209  -> pairs with 125 (.009)

128 is also the FIRST POWER OF 2 in the project and the deepest prime power tested (2^7),
extending C6's prime-power set to p=2. (Z/128)* = Z2 x Z32 is NOT cyclic, so it has no
discrete log: the scalar multiplicative readout must SKIP it loudly, as 119/120/165 already
are, and only the product-character path (C23) applies. Same for 123 and 91.

Dataset size was calibrated against k04's own grok rates before choosing: the transition
sits at ~1,700-2,000 training examples (49 -> 0/6 at 720, 63 -> 2/6 at 1,190, 75 -> 2/3 at
1,687, 81 -> 3/3 at 1,968). The smallest modulus here, n=91, has 2,484 -- above 81's 3/3.

ONE CHANGE from k04 besides the modulus set: max_logit and weight_norm are logged, which
O5 asked for and which k01_scout logged but k02-k08 dropped. They are APPENDED to hist_cols
so columns 0-3 keep their k03/k04 meaning, and max_logit uses k01_scout's exact definition
-- `logits.abs().max()` on the TRAIN batch -- or the numbers are not comparable to the scout
data O5 was read from.

Protocol is otherwise byte-identical to k02 Arm B / k03 Arm B / k04 so all four POOL. Never
pool with the units-only replication arm.

Provenance: __GIT_SHA__ is substituted by run_kernel.py at push time. Kaggle has no git,
so an env-var lookup silently stamps "unknown" -- that is why it is baked in.
"""
import json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "results/k09_primes_zdd"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "no GPU"
cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
assert cap in torch.cuda.get_arch_list(), f"{cap} unsupported by torch {torch.__version__}"
print(f"{torch.cuda.get_device_name(0)} {cap}", flush=True)

GIT_SHA = "__GIT_SHA__"                       # substituted by run_kernel.py
STEPS = 40_000
SNAP_EVERY = 4_000                           # W_E trajectory for the convergence audit
# G1 first (the project has ONE prime), then the zdd-axis fillers most valuable to
# W1: n=128 alone creates 3 of the 5 new pairs. A cut session keeps the top.
MODULI = [127, 131, 128, 123, 91]
SEEDS = [0, 1, 2]
D_MODEL, N_HEADS, D_HEAD, D_MLP = 128, 4, 32, 512
TRAIN_FRAC, LR, WD, BETAS = 0.30, 1e-3, 1.0, (0.9, 0.98)
WALL = 8.0 * 3600
T0 = time.time()
RESULTS = f"{OUT}/k09_summary.json"
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
            # k01_scout's EXACT definitions, or these do not compare to the scout data
            # O5 was read from: abs-max over the TRAIN logits, and the global weight norm.
            mx = logits.abs().max().item()
            wn = sum((p ** 2).sum() for p in m.parameters()).sqrt().item()
            hist.append((step, loss.item(), F.cross_entropy(lo, yte).item(), te, mx, wn))
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
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config=json.dumps(dict(arm=arm, n=n, seed=seed, units_only=units_only,
                               steps=STEPS, train_frac=TRAIN_FRAC, lr=LR, wd=WD,
                               d_model=D_MODEL, engine="pytorch-autograd")))))}
    np.savez_compressed(f"{OUT}/WE_{tag}.npz", **prov, W_E=m.W_E.detach().cpu().numpy(),
                        W_U=m.W_U.detach().cpu().numpy(),
                        hist=np.array(hist),
                        # APPENDED, never inserted: columns 0-3 keep their k03/k04
                        # meaning so an index-based reader cannot silently shift.
                        hist_cols=np.array(["step", "train_loss", "test_loss", "test_acc",
                                            "max_logit", "weight_norm"]),
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
