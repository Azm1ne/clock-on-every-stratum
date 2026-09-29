"""O18 -- the same k03 recipe, on CPU, to find where the engine/torch sparsity gap lives.

  PYTHONPATH=. .venv/bin/python run_o18_cpu.py [n] [seed,seed,...] [steps] [--f64]

`--f64` is O18-F64 (2026-09-15): the SAME kernel source, on CPU, with torch's default
dtype set to float64 -- the one cell of the precision x implementation 2x2 that has
never been run. N9 varied dtype INSIDE the engine (inert there); nothing in this repo
has ever trained torch at anything but float32. Writes `results/o18_f64`.

Seeds run SEQUENTIALLY in one process on purpose: the kernel keeps its resume state in a
single `grid_summary.json` per output directory, so parallel processes sharing that
directory would clobber each other's summary and each other's `seen` set.

C30 said the gap was precision. N9 falsified that (f = -0.07, the sign is wrong). What is
left standing is the specific init draw and the EXECUTION SUBSTRATE: every torch number in
this project was trained on a Kaggle T4; every engine number on this CPU. This runs the
k03 kernel's own training code, unchanged, on the CPU.

  reads ~0.77  -> the difference is GPU-specific, and every Kaggle number inherits it
  reads ~0.55  -> it is in the code, and the two implementations can be bisected line by
                  line, starting from the init draw

WHY THIS EXECS THE KERNEL INSTEAD OF COPYING IT. A copied training loop can drift from the
one that produced the numbers being compared against, silently, and then the A/B is not an
A/B. This applies a SHORT, EXPLICIT list of substitutions to the kernel's own source and
asserts each one matches exactly once, so the diff is the code and cannot rot unnoticed.
Everything not in PATCHES -- architecture, init, optimiser, dataset, split, loss, schedule,
the saved arrays -- is byte-identical to `kernels/k03_grid_acts/run.py`.
"""
import os
import subprocess
import sys
import time

SRC = "kernels/k03_grid_acts/run.py"
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
F64 = "--f64" in sys.argv
N = int(ARGS[0]) if len(ARGS) > 0 else 113
SEEDS = [int(s) for s in ARGS[1].split(",")] if len(ARGS) > 1 else [0, 1, 2]
STEPS = int(ARGS[2]) if len(ARGS) > 2 else 40_000

# (old, new) -- each asserted to occur EXACTLY ONCE in the kernel source.
PATCHES = [
    ('OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "results/k03_grid_acts"',
     'OUT = "results/o18_f64"' if F64 else 'OUT = "results/o18_cpu"'),
    ('DEV = "cuda" if torch.cuda.is_available() else "cpu"',
     'DEV = "cpu"'),
    ('assert torch.cuda.is_available(), "no GPU"\n'
     'cap = "sm_%d%d" % torch.cuda.get_device_capability(0)\n'
     'assert cap in torch.cuda.get_arch_list(), f"{cap} unsupported by torch {torch.__version__}"\n'
     'print(f"{torch.cuda.get_device_name(0)} {cap}", flush=True)\n',
     'print(f"CPU torch {torch.__version__} threads={torch.get_num_threads()}", flush=True)\n'),
    ('STEPS = 40_000', f'STEPS = {STEPS}'),
    # The SUBSTRATE is this experiment's one variable, so it belongs in the artifact.
    # The three runs of 2026-09-14 predate this line; their device is recorded in the
    # pre-registration Outcome and in logs/o18_cpu.log ("CPU torch 2.14.0+cpu threads=6").
    ('d_model=D_MODEL, engine="pytorch-autograd"',
     'd_model=D_MODEL, engine="pytorch-autograd", device=DEV,\n'
     '                               torch_version=torch.__version__,\n'
     '                               dtype=str(torch.get_default_dtype()).replace("torch.", ""),\n'
     '                               threads=torch.get_num_threads()'),
    ('WALL = 7.0 * 3600', 'WALL = 36.0 * 3600'),
    ('jobs = [("A_replication", 113, 0, True, True)]                       # O1 first\n'
     'jobs += [("B_thesis", n, s, False, True) for n in MODULI for s in SEEDS]   # N8: every seed\n',
     'jobs = [("B_thesis", %d, s, False, True) for s in %r]   # O18: k03 cells, on CPU\n'
     % (N, SEEDS)),
]

# The ONE variable of O18-F64. Placed at the import, before any tensor exists, so the
# parameter draw, the forward, the loss and every Adam moment are float64. Everything else
# -- architecture, init scales, optimiser, dataset, split, schedule -- is untouched.
if F64:
    PATCHES.append(("import torch.nn.functional as F",
                    "import torch.nn.functional as F\ntorch.set_default_dtype(torch.float64)"))


def patched_source(src=SRC, patches=PATCHES):
    code = open(src).read()
    for old, new in patches:
        assert code.count(old) == 1, f"patch does not match exactly once:\n{old!r}"
        code = code.replace(old, new)
    return code


def _selfcheck():
    """The patches applied cleanly and touched nothing they should not."""
    code = patched_source()
    assert "cuda" not in code, "a cuda reference survived the patch"
    assert ('OUT = "results/o18_f64"' if F64 else 'OUT = "results/o18_cpu"') in code
    # the one variable, asserted in BOTH directions: a silent no-op patch would make the
    # f64 arm a duplicate of the f32 arm and the 2x2 would read "no effect" for free.
    assert ("torch.set_default_dtype(torch.float64)" in code) == F64, \
        "the float64 patch did not land (or landed when it should not have)"
    # the training loop itself must be untouched
    for line in ('loss = F.cross_entropy(logits, ytr)',
                 'opt.zero_grad(); loss.backward(); opt.step()',
                 'perm = np.random.default_rng(seed).permutation(len(y))',
                 'TRAIN_FRAC, LR, WD, BETAS = 0.30, 1e-3, 1.0, (0.9, 0.98)'):
        assert line in code, f"the patch damaged the recipe: {line!r} is gone"
    print("run_o18_cpu selfcheck PASS")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); sys.exit(0)
    _selfcheck()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True).stdout.strip() != ""
    # Kaggle has no git, so the kernel reads KERNEL_GIT_SHA. Locally we DO have git, and an
    # `unknown` stamp is a defect, not a chore (LAB_PROTOCOL.md constraint 0).
    os.environ["KERNEL_GIT_SHA"] = sha + ("-dirty" if dirty else "")
    out = "results/o18_f64" if F64 else "results/o18_cpu"
    os.makedirs(out, exist_ok=True)
    print(f"O18 n={N} seeds={SEEDS} steps={STEPS} dtype={'float64' if F64 else 'float32'} "
          f"out={out} sha={os.environ['KERNEL_GIT_SHA']}", flush=True)
    t0 = time.time()
    g = {"__name__": "__main__", "__file__": SRC}
    exec(compile(patched_source(), SRC, "exec"), g)
    print(f"O18 done in {(time.time()-t0)/60:.1f} min", flush=True)
