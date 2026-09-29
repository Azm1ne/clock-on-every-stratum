"""O18-T -- is the engine/torch gap the INIT DRAW or the UPDATE PATH?

  PYTHONPATH=. .venv/bin/python run_o18_transplant.py --stage1 [n] [seed] [steps]
  PYTHONPATH=. .venv/bin/python run_o18_transplant.py --selfcheck

Criteria fixed in experiments/PREREGISTER_o18_transplant.md, committed BEFORE this ran.

Transplant torch's nine sampled init tensors into the from-scratch engine, then step both
on IDENTICAL batches and compare every parameter after every step. If the two update paths
agree to rounding, the gap cannot live there -- established in minutes instead of the ~5 h
a 40k A/B costs at 449 ms/step.

WHY THIS EXECS THE KERNEL'S OWN SOURCE (same discipline as run_o18_cpu.py): a copied init
can drift from the one that produced the numbers we are comparing against, silently. A
short explicit substitution list is applied and each patch is asserted to match EXACTLY
ONCE, so the diff is the code and cannot rot unnoticed. `jobs = []` stops the module before
it trains anything; everything else -- architecture, init, dataset, split -- is byte-identical
to kernels/k03_grid_acts/run.py.

LAYOUT IS NEVER REASONED ABOUT. The engine holds W_Q/W_K/W_V as (d_model, H*d_head) and
W_O as (H*d_head, d_model); the kernel uses (H, d_model, d_head) and (H, d_head, d_model).
Three bin-convention arguments have been lost to reasoning in this project, so the mapping
is a GATE asserted by logit equality (T0) and the run refuses to start if it fails.
"""
import json, os, sys, time

ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
N = int(ARGS[0]) if ARGS else 113
SEED = int(ARGS[1]) if len(ARGS) > 1 else 0
STEPS = int(ARGS[2]) if len(ARGS) > 2 else 500
OUT = "results/o18_transplant"

import numpy as np
import torch
import torch.nn.functional as F

SRC = "kernels/k03_grid_acts/run.py"
PATCHES = [
    ('DEV = "cuda" if torch.cuda.is_available() else "cpu"', 'DEV = "cpu"'),
    ('assert torch.cuda.is_available(), "no GPU"\n'
     'cap = "sm_%d%d" % torch.cuda.get_device_capability(0)\n'
     'assert cap in torch.cuda.get_arch_list(), f"{cap} unsupported by torch {torch.__version__}"\n'
     'print(f"{torch.cuda.get_device_name(0)} {cap}", flush=True)\n', ''),
    ('OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "results/k03_grid_acts"',
     f'OUT = "{OUT}"'),
    ('jobs = [("A_replication", 113, 0, True, True)]                       # O1 first\n'
     'jobs += [("B_thesis", n, s, False, True) for n in MODULI for s in SEEDS]   # N8: every seed\n',
     'jobs = []   # O18-T: import the kernel\'s init and dataset; train nothing here\n'),
]


def kernel_ns():
    """The kernel's own Transformer/dataset, with training disabled. Patches asserted."""
    src = open(SRC).read()
    for old, new in PATCHES:
        assert src.count(old) == 1, f"patch matched {src.count(old)}x, expected 1:\n{old[:90]}"
        src = src.replace(old, new)
    ns = {"__name__": "k03_patched", "__file__": SRC}
    exec(compile(src, SRC, "exec"), ns)
    return ns


# ---------------------------------------------------------------- the transplant

def to_engine(t, name, H, DH, D):
    """One torch tensor -> the engine's layout. Asserted by T0, never by argument."""
    a = t.detach().cpu().numpy().astype(np.float64)
    if name in ("W_Q", "W_K", "W_V"):        # (H, D, DH) -> (D, H*DH)
        return a.transpose(1, 0, 2).reshape(D, H * DH)
    if name == "W_O":                        # (H, DH, D) -> (H*DH, D)
        return a.reshape(H * DH, D)
    return a                                 # W_E, W_pos, W_in, W_out, W_U: same shape


def transplant(tm, em, H, DH, D):
    moved = []
    for name in ("W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U"):
        src = getattr(tm, name)
        dst = getattr(em, name)
        w = to_engine(src, name, H, DH, D)
        assert w.shape == dst.data.shape, f"{name}: {w.shape} vs engine {dst.data.shape}"
        dst.data[...] = w.astype(dst.data.dtype)
        moved.append(name)
    return moved


def check_transplant(tm, em, toks, tol=1e-10):
    """T0 GATE. Same weights must compute the same function, or nothing downstream reads."""
    # The kernel samples with torch.randn, so tm is FLOAT32 and its forward accumulates
    # float32 rounding -- which reads as 4e-7 against a float64 engine and looks exactly
    # like a layout bug. The criterion says "at float64", so the torch side is promoted
    # for the check. float32 -> float64 -> float32 is exact, so tm is unchanged after.
    with torch.no_grad():
        lt = tm.double()(torch.as_tensor(toks).long()).numpy()
        tm.float()
    from src.autograd.engine import no_grad
    with no_grad():
        le = np.asarray(em(toks).data, dtype=np.float64)
    d = float(np.abs(lt - le).max())
    scale = float(np.abs(lt).max())
    return d, scale, d < tol


# ---------------------------------------------------------------- stage 1

def _reimport(dtype):
    """DTYPE is read at import time (src/autograd/engine.py), so two dtypes cannot be live
    in one process. The arms therefore run SEQUENTIALLY and are compared from snapshots."""
    os.environ["ENGINE_DTYPE"] = dtype
    for mod in [m for m in list(sys.modules) if m.startswith("src.")]:
        del sys.modules[mod]


def _snap(named, tm, H, DH, D):
    """Parameters as float64 numpy in the ENGINE's layout, so all three arms compare
    directly regardless of which framework produced them."""
    if named is not None:
        return {k: np.asarray(p.data, dtype=np.float64) for k, p in named}
    return {k: to_engine(getattr(tm, k), k, H, DH, D)
            for k in ("W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U")}


def torch_arm(ns, steps, marks):
    D, H, DH = ns["D_MODEL"], ns["N_HEADS"], ns["D_HEAD"]
    xtr, ytr, xte, yte, toks_all, y_all = ns["dataset"](N, SEED, False)
    tm = ns["Transformer"](N + 1, N, SEED).to("cpu")
    opt = torch.optim.AdamW(tm.parameters(), lr=ns["LR"], weight_decay=ns["WD"],
                            betas=ns["BETAS"])
    snaps, losses = {}, {}
    for t in range(1, steps + 1):
        loss = F.cross_entropy(tm(xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        if t in marks:
            snaps[t] = _snap(None, tm, H, DH, D)
            losses[t] = float(loss.item())
    return snaps, losses, (xtr.cpu().numpy(), ytr.cpu().numpy(), toks_all)


def engine_arm(ns, dtype, steps, marks, data):
    """One engine arm at one dtype, transplanted from torch's own sampled init."""
    _reimport(dtype)
    from src.model.transformer import Transformer as ETransformer
    from src.autograd.nn import AdamW as EAdamW, cross_entropy
    D, H, DH, DM = ns["D_MODEL"], ns["N_HEADS"], ns["D_HEAD"], ns["D_MLP"]
    xtr_np, ytr_np, toks_all = data
    tm = ns["Transformer"](N + 1, N, SEED).to("cpu")          # same seed => same weights
    em = ETransformer(N + 1, N, d_model=D, n_heads=H, d_head=DH, d_mlp=DM, seed=SEED)
    transplant(tm, em, H, DH, D)
    gate = check_transplant(tm, em, toks_all) if dtype == "float64" else None
    named = [(k, v) for k, v in vars(em).items()
             if hasattr(v, "requires_grad") and v.requires_grad]
    opt = EAdamW(em.parameters(), lr=ns["LR"], betas=ns["BETAS"], weight_decay=ns["WD"])
    snaps, losses = {}, {}
    for t in range(1, steps + 1):
        loss = cross_entropy(em(xtr_np), ytr_np)
        opt.zero_grad(); loss.backward(); opt.step()
        if t in marks:
            snaps[t] = _snap(named, None, H, DH, D)
            losses[t] = float(np.asarray(loss.data).ravel()[0])
    return snaps, losses, gate


def compare(a, b):
    """Pre-registered metric: max over params/elements of |d| / (|b| + 1e-12). Also a
    per-tensor max|d|/rms -- ADDED, because the pre-registered form is dominated by the
    elements where the reference weight is ~0 and swings orders of magnitude per step."""
    worst, per = 0.0, {}
    for k in a:
        d = np.abs(a[k] - b[k])
        rms = float(np.sqrt((b[k] ** 2).mean()))
        per[k] = dict(rel=float((d / (np.abs(b[k]) + 1e-12)).max()), absmax=float(d.max()),
                      norm=float(d.max() / rms) if rms else float("nan"))
        worst = max(worst, per[k]["rel"])
    return worst, per


def deterministic_check(ns, steps=3):
    """ADDED, not pre-registered -- labelled so in the Outcome.

    500 float32 steps can only rule out a GROSS difference in the update path; float32
    chaos swamps a subtle systematic one. This runs both arms at FLOAT64 from the same
    transplanted weights and compares, at each step, first the GRADIENTS (the autograd
    graph) and then the WEIGHTS AFTER the optimiser step (the optimiser itself). With no
    float32 rounding there is no chaos to hide in: agreement at ~1e-14 means the update
    path is identical, not merely indistinguishable.

    C10 verified gradients to 2.3e-15, but against a torch reference written in the
    ENGINE's layout -- not against the kernel that actually produced the torch numbers.
    This closes that gap."""
    _reimport("float64")
    from src.model.transformer import Transformer as ETransformer
    from src.autograd.nn import AdamW as EAdamW, cross_entropy
    D, H, DH, DM = ns["D_MODEL"], ns["N_HEADS"], ns["D_HEAD"], ns["D_MLP"]
    xtr, ytr, xte, yte, toks_all, y_all = ns["dataset"](N, SEED, False)
    xtr_np, ytr_np = xtr.cpu().numpy(), ytr.cpu().numpy()

    tm = ns["Transformer"](N + 1, N, SEED).to("cpu").double()
    em = ETransformer(N + 1, N, d_model=D, n_heads=H, d_head=DH, d_mlp=DM, seed=SEED)
    transplant(tm, em, H, DH, D)
    named = [(k, v) for k, v in vars(em).items()
             if hasattr(v, "requires_grad") and v.requires_grad]
    topt = torch.optim.AdamW(tm.parameters(), lr=ns["LR"], weight_decay=ns["WD"],
                             betas=ns["BETAS"])
    eopt = EAdamW(em.parameters(), lr=ns["LR"], betas=ns["BETAS"], weight_decay=ns["WD"])

    print("\n" + "=" * 92)
    print("ADDED (not pre-registered): deterministic float64 single-step comparison")
    print("=" * 92)
    print(f"  {'step':>5} {'max|dloss|':>12} {'max|dgrad|':>12} {'grad tensor':>12} "
          f"{'max|dw| after opt':>18} {'w tensor':>9}")
    out = []
    for t in range(1, steps + 1):
        lo = tm(xtr)
        lt = F.cross_entropy(lo, ytr)
        topt.zero_grad(); lt.backward()
        le = cross_entropy(em(xtr_np), ytr_np)
        eopt.zero_grad(); le.backward()

        gd = {}
        for k, p in named:
            tg = to_engine(getattr(tm, k).grad, k, H, DH, D)
            gd[k] = float(np.abs(np.asarray(p.grad, dtype=np.float64) - tg).max())
        gw = max(gd, key=gd.get)

        topt.step(); eopt.step()
        wd_ = {}
        for k, p in named:
            tw = to_engine(getattr(tm, k), k, H, DH, D)
            wd_[k] = float(np.abs(np.asarray(p.data, dtype=np.float64) - tw).max())
        ww = max(wd_, key=wd_.get)
        dl = abs(float(lt.item()) - float(np.asarray(le.data).ravel()[0]))
        print(f"  {t:5d} {dl:12.3e} {gd[gw]:12.3e} {gw:>12} {wd_[ww]:18.3e} {ww:>9}")
        out.append(dict(step=t, dloss=dl, dgrad=gd, dw=wd_))
    print("\n  Interpretation: gradients agreeing at ~1e-14 means the autograd GRAPH is the")
    print("  same; weights agreeing after the step means the OPTIMISER is the same too.")
    return out


def stage1(ns):
    os.makedirs(OUT, exist_ok=True)
    print("=" * 92)
    print(f"O18-T STAGE 1 -- paired update-path comparison, n={N} seed={SEED}, {STEPS} steps")
    print("criteria: experiments/PREREGISTER_o18_transplant.md")
    print("=" * 92)
    marks = sorted({1, 2, 3, 4, 5} | {t for t in range(0, STEPS + 1, max(1, STEPS // 10)) if t}
                   | {STEPS})

    t_snap, t_loss, data = torch_arm(ns, STEPS, marks)
    xtr_np, ytr_np, _ = data
    from src.train.loop import modular_data
    extr, eytr, exte, eyte = modular_data(N, "mul", train_frac=ns["TRAIN_FRAC"], seed=SEED)
    t1 = np.array_equal(extr, xtr_np) and np.array_equal(eytr, ytr_np)
    print(f"\nT1 split byte-identical (engine modular_data vs kernel dataset): "
          f"{'HELD' if t1 else 'NOT HELD'}   train {xtr_np.shape[0]}")
    assert t1, "T1 failed: the arms are not on the same data, nothing else is comparable"

    e64, l64, gate = engine_arm(ns, "float64", STEPS, marks, data)
    d, scale, ok = gate
    print(f"T0 GATE transplant exact: max |dlogit| = {d:.3e} (logit scale {scale:.3f})  "
          f"{'HELD' if ok else 'NOT HELD'}")
    if not ok:
        print("\n  T0 FAILED -- the layout map is wrong, the two models do not compute the")
        print("  same function, and NOTHING downstream is readable. Not starting.")
        return None
    e32, l32, _ = engine_arm(ns, "float32", STEPS, marks, data)
    det = deterministic_check(ns)

    print("\n  T2 PRIMARY : engine float32 vs TORCH  float32   (the update-path question)")
    print("  T3 BASELINE: engine float32 vs ENGINE float64   (same code, dtype only)")
    print(f"\n  {'step':>6} | {'T2 r':>10} {'T2 max|d|':>11} {'worst':>7} | "
          f"{'T3 r':>10} {'T3 max|d|':>11} | {'ratio':>9}")
    rows = []
    for t in marks:
        r2, p2 = compare(e32[t], t_snap[t])
        r3, p3 = compare(e32[t], e64[t])
        w = max(p2, key=lambda k: p2[k]["norm"])
        w3 = max(p3, key=lambda k: p3[k]["norm"])
        ratio = r2 / r3 if r3 > 0 else float("inf")
        rows.append(dict(step=t, r_t2=r2, r_t3=r3, ratio=ratio, worst=w, t2=p2, t3=p3,
                         loss_t=t_loss[t], loss_e32=l32[t], loss_e64=l64[t]))
        print(f"  {t:6d} | {r2:10.3e} {p2[w]['absmax']:11.3e} {w:>7} | "
              f"{r3:10.3e} {p3[w3]['absmax']:11.3e} | {ratio:9.2f}")

    fin = rows[-1]
    print("\n" + "=" * 92 + "\nPRE-REGISTERED DECISION (T2)\n" + "=" * 92)
    print(f"  T2  engine-f32 vs torch-f32   at step {STEPS}: {fin['r_t2']:.3e}")
    print(f"  T3  engine-f32 vs engine-f64  at step {STEPS}: {fin['r_t3']:.3e}  <- rounding")
    print(f"  ratio = {fin['ratio']:.2f}    rule: < 100 => update paths agree to rounding")
    agree = fin["ratio"] < 100
    if agree:
        print("\n  => UPDATE PATHS AGREE TO ROUNDING. Hypothesis FALSIFIED. The gap is the")
        print("     init DRAW, or something not yet enumerated. Stage 2 is NOT run.")
    else:
        first = next((r for r in rows if r["ratio"] >= 100), fin)
        print(f"\n  => DIVERGENCE IS REAL. First step over the rule: {first['step']}, "
              f"carried by {first['worst']}. Run stage 2.")
    res = dict(n=N, seed=SEED, steps=STEPS, t0_dlogit=d, t1_split=bool(t1),
               verdict="agree" if agree else "diverge", ratio=fin["ratio"],
               rows=[{k: v for k, v in r.items() if k not in ("t2", "t3")} for r in rows],
               per_tensor_final=dict(t2=fin["t2"], t3=fin["t3"]),
               deterministic_float64=det)
    from src import provenance
    res["provenance"] = provenance.stamp(dict(experiment="O18-T", n=N, seed=SEED,
                                              steps=STEPS, engine="from-scratch+torch"))
    with open(f"{OUT}/stage1.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\n  wrote {OUT}/stage1.json")
    print("  Fill the Outcome of experiments/PREREGISTER_o18_transplant.md; never edit above it.")
    return res


def stage2(ns, steps=40_000, snap_every=1_000):
    """The 40k A/B, engine trained FROM TORCH'S OWN SAMPLED INIT.

    DEVIATION from the pre-registration, recorded in its Outcome: the file says stage 2 is
    not run when stage 1 shows agreement. That conditional was written on the reasoning
    that 'which endpoint' is only meaningful once divergence is established. Stage 1
    disproved the reasoning, not the value: with the update path identical to 1e-15 and the
    gap shown to live in the WEIGHTS rather than the instrument, the endpoint question is
    now the DIRECT test of the last enumerated candidate -- the init draw. T4's f-rule is
    unchanged; only the gate on reaching it is.

    Output is k03's schema so analyze_n7.py / analyze_n9.py consume it unchanged."""
    _reimport("float64")
    from src.model.transformer import Transformer as ETransformer
    from src.autograd.nn import AdamW as EAdamW, cross_entropy
    from src.autograd.engine import no_grad, DTYPE
    from src.provenance import stamp_npz
    os.makedirs(OUT, exist_ok=True)
    D, H, DH, DM = ns["D_MODEL"], ns["N_HEADS"], ns["D_HEAD"], ns["D_MLP"]
    xtr, ytr, xte, yte, toks_all, y_all = ns["dataset"](N, SEED, False)
    xtr_np, ytr_np = xtr.cpu().numpy(), ytr.cpu().numpy()
    xte_np, yte_np = xte.cpu().numpy(), yte.cpu().numpy()

    tm = ns["Transformer"](N + 1, N, SEED).to("cpu")
    em = ETransformer(N + 1, N, d_model=D, n_heads=H, d_head=DH, d_mlp=DM, seed=SEED)
    transplant(tm, em, H, DH, D)
    d, scale, ok = check_transplant(tm, em, toks_all)
    assert ok, f"T0 gate failed at stage 2: max |dlogit| {d:.3e}"
    print(f"stage 2: T0 gate {d:.3e}; engine dtype {DTYPE}; {steps} steps", flush=True)
    opt = EAdamW(em.parameters(), lr=ns["LR"], betas=ns["BETAS"], weight_decay=ns["WD"])

    hist, traj, traj_steps, grok = [], [], [], None
    t0 = time.time()
    for t in range(steps + 1):
        loss = cross_entropy(em(xtr_np), ytr_np)
        opt.zero_grad(); loss.backward(); opt.step()
        if t % 200 == 0:
            with no_grad():                       # ONE forward; the test set is 2.3x train
                out = em(xte_np)
                acc = float((np.asarray(out.data).argmax(-1) == yte_np).mean())
                tel = float(np.asarray(cross_entropy(out, yte_np).data).ravel()[0])
            hist.append((t, float(np.asarray(loss.data).ravel()[0]), tel, acc))
            if grok is None and acc > 0.99:
                grok = t
        if t % snap_every == 0:
            traj.append(np.asarray(em.W_E.data, dtype=np.float32))
            traj_steps.append(t)
        if t % 2000 == 0:
            print(f"  step {t} acc {hist[-1][3]:.4f} [{(time.time()-t0)/60:.0f}m]", flush=True)
    tag = f"WE_transplant_n{N}_s{SEED}"
    np.savez_compressed(
        f"{OUT}/{tag}.npz",
        **stamp_npz(dict(experiment="O18-T-stage2", arm="engine-from-torch-init", n=N,
                         seed=SEED, steps=steps, train_frac=ns["TRAIN_FRAC"], lr=ns["LR"],
                         wd=ns["WD"], d_model=D, engine="from-scratch", dtype=str(DTYPE),
                         init_source="kernels/k03_grid_acts/run.py::Transformer",
                         grok_step=grok)),
        W_E=np.asarray(em.W_E.data), W_U=np.asarray(em.W_U.data),
        hist=np.array(hist),
        hist_cols=np.array(["step", "train_loss", "test_loss", "test_acc"]),
        we_traj=np.stack(traj), we_traj_steps=np.array(traj_steps), y_all=y_all)
    print(f"stage 2 done in {(time.time()-t0)/60:.0f} min; grok {grok}; wrote {OUT}/{tag}.npz",
          flush=True)


def _selfcheck():
    ns = kernel_ns()
    assert ns["D_MODEL"] == 128 and ns["N_HEADS"] == 4 and ns["D_HEAD"] == 32
    assert ns["LR"] == 1e-3 and ns["WD"] == 1.0 and ns["TRAIN_FRAC"] == 0.30
    assert ns["jobs"] == [], "the kernel must not have trained during import"
    D, H, DH = ns["D_MODEL"], ns["N_HEADS"], ns["D_HEAD"]
    # the layout map must be a PERMUTATION -- no element created or destroyed
    for name, sh in (("W_Q", (H, D, DH)), ("W_O", (H, DH, D))):
        a = torch.arange(int(np.prod(sh))).reshape(sh).double()
        b = to_engine(a, name, H, DH, D)
        assert sorted(b.ravel().tolist()) == sorted(a.numpy().ravel().tolist())
    # and the T0 gate must FAIL on a deliberately wrong map, or it is not a gate
    a = torch.randn(H, D, DH, generator=torch.Generator().manual_seed(0)).double()
    good = to_engine(a, "W_Q", H, DH, D)
    bad = a.numpy().reshape(D, H * DH)                      # the plausible-but-wrong order
    assert not np.allclose(good, bad), "the wrong map is indistinguishable -- T0 cannot bite"
    print("test o18_transplant selfcheck PASS -- kernel patches match once, module trains "
          "nothing, layout map is a permutation, and the wrong map is distinguishable")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    elif "--stage2" in sys.argv:
        stage2(kernel_ns(), steps=STEPS if STEPS > 1000 else 40_000)
    else:
        stage1(kernel_ns())
