"""N7 -- T2 on the FROM-SCRATCH ENGINE. Zero Kaggle quota, local CPU.

Pre-registered in experiments/PREREGISTER_n7_engine.md, committed before this ran.
Constraint 2: anything trained on PyTorch autograd is scout data. This is the engine, so
what it produces is paper data.

One run = one process = one npz, so seeds 3-4 can be added later as an independent wave
without re-running anything. Resumable: a reboot costs <=2,500 steps, not the night.

Output schema is k03's exactly (WE_<tag>.npz), so analyze_n4.py, render_all.py and
scripts/look.py consume it unchanged -- none of them needed a line changed for this.

  PYTHONPATH=. OMP_NUM_THREADS=2 .venv/bin/python run_n7.py <n> <seed> [steps]
"""
import json, os, sys, time
import numpy as np
from src.model.transformer import Transformer
from src.train.loop import modular_data      # same split RNG as k03's dataset(): E5 is paired
from src.autograd.engine import Tensor, no_grad, DTYPE
from src.autograd.nn import cross_entropy, AdamW
from src.provenance import stamp_npz

N, SEED = int(sys.argv[1]), int(sys.argv[2])
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 40_000
LOG, CKPT, TRAJ, SNAP = 100, 2_500, 1_000, 5_000
# train_frac is 0.30 for every real run (plan 3.4). The override exists so the smoke
# test can use >=0.8 at small n, where 0.30 is below critical dataset size (C14) and a
# correct implementation would look broken -- and so the grok branch gets exercised.
TRAIN_FRAC = float(os.environ.get("N7_TRAIN_FRAC", 0.30))
LR, WD, BETAS = 1e-3, 1.0, (0.9, 0.98)
D_MODEL, N_HEADS, D_HEAD, D_MLP = 128, 4, 32, 512
OUT = os.environ.get("N7_OUT", "results/n7_engine")
TAG = f"engine_n{N}_s{SEED}"
RES, CK = f"{OUT}/WE_{TAG}.npz", f"{OUT}/ckpt_{TAG}.npz"   # ckpt_ must NOT match WE_*
os.makedirs(OUT, exist_ok=True)

# Six of these run at once; each snapshot builds full-grid activations, so a simultaneous
# spike across all six is the realistic OOM path on a 14 GB box. Stagger them.
SNAP_OFF = (SEED * 1000 + (N % 4) * 250) % SNAP


def save_atomic(path, compress=True, **arrs):
    """A torn file at hour 12 is indistinguishable from no file at all. Write, then rename."""
    tmp = path + ".tmp.npz"                     # already ends .npz, so savez appends nothing
    (np.savez_compressed if compress else np.savez)(tmp, **arrs)
    os.replace(tmp, path)


def named_params(m):
    return [(k, v) for k, v in vars(m).items() if isinstance(v, Tensor) and v.requires_grad]


xtr, ytr, xte, yte = modular_data(N, "mul", train_frac=TRAIN_FRAC, seed=SEED)
a, b = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
x_all = np.stack([a.ravel(), b.ravel(), np.full(N * N, N)], 1)
y_all = (a.ravel() * b.ravel()) % N

m = Transformer(N + 1, N, d_model=D_MODEL, n_heads=N_HEADS, d_head=D_HEAD, d_mlp=D_MLP,
                seed=SEED)
opt = AdamW(m.parameters(), lr=LR, betas=BETAS, weight_decay=WD)
# parameters() is a vars() comprehension, so name order == optimizer state order. If that
# ever stops being true, resume would silently restore moments onto the wrong tensors.
assert [id(v) for _, v in named_params(m)] == [id(p) for p in opt.params], \
    "parameter order diverged from named_params -- resume would corrupt optimizer state"

CFG = dict(task="mul", n=N, steps=STEPS, train_frac=TRAIN_FRAC, d_model=D_MODEL,
           n_heads=N_HEADS, d_head=D_HEAD, d_mlp=D_MLP, lr=LR, wd=WD, betas=BETAS,
           seed=SEED, engine="from-scratch", arm="engine", experiment="N7",
           # TRAINING PRECISION IS AN EXPERIMENTAL VARIABLE (C30/O17), so it is stamped
           # like any other. Pre-N7 artifacts carry no `dtype` key: absent means float64
           # on the engine and float32 on every Kaggle/PyTorch kernel.
           dtype=str(DTYPE))

# The runner deletes CK on a clean finish, so "result present, no resume state" means
# done. Guard here rather than in the launcher: every invocation path routes through this,
# and re-running the sweep must never silently overwrite a finished 18-hour run.
if os.path.exists(RES) and not os.path.exists(CK):
    _z = np.load(RES, allow_pickle=False)
    if int(_z["step"]) >= STEPS:
        print(f"{TAG}: already complete at step {int(_z['step'])} -- nothing to do", flush=True)
        sys.exit(0)

start, hist, grok, traj, traj_steps = 0, [], None, [], []
if os.path.exists(CK):
    z = np.load(CK, allow_pickle=False)
    for k, p in named_params(m):
        p.data[...] = z["p_" + k]
    opt.m = [z["m_" + k] for k, _ in named_params(m)]
    opt.v = [z["v_" + k] for k, _ in named_params(m)]
    opt.t = int(z["t"])
    start = int(z["step"]) + 1
    hist = json.loads(str(z["hist_json"]))
    grok = int(z["grok"]) if int(z["grok"]) >= 0 else None
    traj, traj_steps = list(z["traj"]), list(z["traj_steps"])
    print(f"resumed {TAG} at step {start} (grok={grok}, {len(traj)} traj points)", flush=True)

HCOLS = ["step", "train_loss", "test_loss", "train_acc", "test_acc"]


def snapshot(step, extra_prov=None):
    """Full-grid activations. Chunked and under no_grad: the grid is 3.3x the training
    batch and needs no graph -- building one here is what took RSS to 10 GB on Gate 1."""
    lgs, pres, atts = [], [], []
    with no_grad():
        for i in range(0, len(x_all), 2000):
            l, p_, at = m(x_all[i:i + 2000], return_acts=True)
            lgs.append(l.data.astype(np.float32))
            pres.append(np.maximum(p_.data, 0).astype(np.float32))
            atts.append(at.data[:, :, -1, :].astype(np.float32))
    save_atomic(RES, **stamp_npz(dict(CFG, at_step=step, **(extra_prov or {}))),
                # EVERY parameter, not just the two the spectral analyses read. The
                # runner deletes CK on a clean finish, so before this line NO completed
                # run in the project had a recoverable model: 0 of 232 artifacts carried
                # W_Q/W_K/W_V/W_O/W_in/W_out/W_pos, and any intervention that runs the
                # forward pass -- rather than post-processing saved logits -- was
                # impossible without retraining. Named `p_*` to match `checkpoint()`, so
                # one loader reads both. Additive: every existing reader looks keys up by
                # name, and ~1.6 MB against a ~26 MB artifact.
                **{f"p_{k}": q.data for k, q in named_params(m)},
                W_E=m.W_E.data, W_U=m.W_U.data,
                logits_all=np.concatenate(lgs), mlp_acts=np.concatenate(pres),
                attn=np.concatenate(atts), y_all=y_all, step=step,
                hist=np.array([[h[c] for c in HCOLS] for h in hist], dtype=float),
                hist_cols=np.array(HCOLS),
                we_traj=np.stack(traj) if traj else np.zeros((0, N + 1, D_MODEL), np.float32),
                we_traj_steps=np.array(traj_steps, dtype=int))


def checkpoint(step):
    st = {f"p_{k}": p.data for k, p in named_params(m)}
    st |= {f"m_{k}": opt.m[i] for i, (k, _) in enumerate(named_params(m))}
    st |= {f"v_{k}": opt.v[i] for i, (k, _) in enumerate(named_params(m))}
    save_atomic(CK, compress=False, step=step, t=opt.t,
                grok=-1 if grok is None else grok, hist_json=json.dumps(hist),
                traj=np.stack(traj) if traj else np.zeros((0, N + 1, D_MODEL), np.float32),
                traj_steps=np.array(traj_steps, dtype=int), **st)


t0 = time.time()
print(f"{TAG}: {len(ytr)} train / {len(yte)} test | "
      f"{sum(p.data.size for p in opt.params):,} params | steps {start}..{STEPS}", flush=True)

for step in range(start, STEPS + 1):
    opt.zero_grad()
    logits = m(xtr)
    loss = cross_entropy(logits, ytr)
    loss.backward()
    opt.step()

    if step % TRAJ == 0:                       # cheap: W_E only, ~64 KB. Feeds E2's tail mean.
        traj.append(m.W_E.data.astype(np.float32).copy())
        traj_steps.append(step)

    if step % LOG == 0:
        with no_grad():
            lte = m(xte)
            te_acc = float((lte.data.argmax(-1) == yte).mean())
            rec = dict(step=step, train_loss=float(loss.data),
                       test_loss=float(cross_entropy(lte, yte).data),
                       train_acc=float((logits.data.argmax(-1) == ytr).mean()),
                       test_acc=te_acc)
        hist.append(rec)
        if grok is None and te_acc > 0.99:
            grok = step
            print(f"  *** {TAG} GROKKED at step {step} ***", flush=True)
            snapshot(step, {"grok_step": step})
        if step % 2000 == 0:
            el = time.time() - t0
            done = max(step - start, 1)
            print(f"  {TAG} step {step:6d} tr {rec['train_acc']:.3f} te {te_acc:.3f} "
                  f"trloss {rec['train_loss']:.2e} [{el/60:.0f}m, "
                  f"eta {el/done*(STEPS-step)/3600:.1f}h]", flush=True)

    if step % SNAP == SNAP_OFF and step > start:
        snapshot(step)
    if step % CKPT == 0 and step > start:
        checkpoint(step)

snapshot(STEPS, {"grok_step": -1 if grok is None else grok})
if os.path.exists(CK):
    os.remove(CK)                              # run finished; resume state is dead weight
print(f"{TAG} done. grok_step={grok}, {(time.time()-t0)/3600:.2f} h", flush=True)
