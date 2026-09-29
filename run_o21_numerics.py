"""O21 follow-up: is the engine's float32 arithmetic better-rounded than torch's?

Pre-registered: experiments/PREREGISTER_o21_dtype.md, AMENDMENT section, criteria A1-A3.

The probe already killed candidate (c): the engine computes in float32 end to end, 0
promoted sites, positive control at 100%. The remaining candidates are (a) matmul
accumulation order and (b) the log-softmax formulation. Both predict the SAME observable --
that numpy at float32 lands nearer the float64 truth than torch at float32 does.

Every float32 result is compared against a float64 reference built by promoting the SAME
float32 inputs. The float32 -> float64 -> float32 round trip is exact, so the inputs are
unchanged and any difference is arithmetic, not representation. (LAB_PROTOCOL.md: a float32
forward compared against a float64 one reads ~1e-7 and looks exactly like a layout bug.)

Usage:  PYTHONPATH=. .venv/bin/python run_o21_numerics.py [--selfcheck]
"""
import os
import sys

os.environ.setdefault("ENGINE_DTYPE", "float32")

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
import torch.nn.functional as F                                      # noqa: E402
from src.model.transformer import Transformer                        # noqa: E402
from src.train.loop import modular_data                              # noqa: E402


def rel_err(x32, ref64):
    """Relative error of a float32 result against the float64 ground truth."""
    a = np.asarray(x32, dtype=np.float64)
    r = np.asarray(ref64, dtype=np.float64)
    denom = np.abs(r).max()
    return float(np.abs(a - r).max() / denom) if denom else 0.0


def verdict(name, e_np, e_t, candidate):
    """Ratio < 0.5 supports the candidate; [0.5, 2.0] is criterion A3."""
    ratio = e_np / e_t if e_t else float("inf")
    print(f"\n  {name}")
    print(f"    numpy  float32 vs float64 truth : {e_np:.3e}")
    print(f"    torch  float32 vs float64 truth : {e_t:.3e}")
    print(f"    ratio numpy/torch               : {ratio:.3f}")
    if ratio <= 0.5:
        print(f"    => {candidate} SUPPORTED: numpy is >=2x better-rounded")
    elif ratio >= 2.0:
        print(f"    => REVERSED: torch is >=2x better-rounded. {candidate} cannot explain O21")
    else:
        print(f"    => within [0.5, 2.0]: numerically equivalent. {candidate} is DEAD")
    return ratio


def matmul_case(a32, b32, tag):
    ref = np.asarray(a32, np.float64) @ np.asarray(b32, np.float64)
    np32 = a32 @ b32
    t32 = (torch.from_numpy(a32) @ torch.from_numpy(b32)).numpy()
    assert np32.dtype == np.float32 and t32.dtype == np.float32, "inputs must stay float32"
    return verdict(f"matmul {tag}  shape {a32.shape} @ {b32.shape}",
                   rel_err(np32, ref), rel_err(t32, ref), "candidate (a) accumulation order")


def logsoftmax_case(x32, tag):
    """Engine formulation: z = x - max(x); z - log(sum(exp(z))). Torch: F.log_softmax."""
    def engine_ls(x):
        z = x - x.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    ref = engine_ls(np.asarray(x32, np.float64))
    np32 = engine_ls(x32)
    t32 = F.log_softmax(torch.from_numpy(x32), dim=-1).numpy()
    assert np32.dtype == np.float32 and t32.dtype == np.float32
    return verdict(f"log_softmax {tag}  shape {x32.shape}",
                   rel_err(np32, ref), rel_err(t32, ref), "candidate (b) softmax formulation")


def _selfcheck():
    """A deliberately bad float32 sum must read WORSE than a good one."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100_000).astype(np.float32)
    ref = np.asarray(x, np.float64).sum()
    good = float(np.asarray(x).sum(dtype=np.float64))      # wide accumulation
    bad = np.float32(0.0)
    for v in x[:1000]:                                     # naive f32 running sum
        bad = np.float32(bad + v)
    e_good = abs(good - ref) / abs(ref)
    e_bad = abs(float(bad) - float(np.asarray(x[:1000], np.float64).sum())) / \
        abs(float(np.asarray(x[:1000], np.float64).sum()))
    assert e_good < e_bad, f"instrument cannot rank accuracy: {e_good:.2e} vs {e_bad:.2e}"
    assert rel_err(np.float32([1.0]), np.float64([1.0])) == 0.0, "identical must read 0"
    print(f"selfcheck OK: wide accumulation {e_good:.2e} ranks better than naive f32 "
          f"{e_bad:.2e}; identical inputs read exactly 0")


def main():
    n, seed = 113, 0
    print("=== O21 follow-up · is numpy float32 better-rounded than torch float32? ===")
    print(f"numpy {np.__version__} | torch {torch.__version__}\n")

    rng = np.random.default_rng(seed)
    ratios = []

    # Synthetic, at the model's real shapes: (3830, 128) @ (128, 512) is the MLP matmul.
    a = rng.standard_normal((3830, 128)).astype(np.float32)
    b = rng.standard_normal((128, 512)).astype(np.float32)
    ratios.append(("matmul/synthetic", matmul_case(a, b, "synthetic")))

    # Real activations: the question is about THIS model's operating regime.
    xtr, ytr, _, _ = modular_data(n, "mul", train_frac=0.30, seed=seed)
    m = Transformer(n_vocab=n + 1, n_out=n, seed=seed)
    logits, pre, _ = m(xtr, return_acts=True)
    real_logits = np.asarray(logits.data, dtype=np.float32)
    real_pre = np.asarray(pre.data, dtype=np.float32)
    W_out = np.asarray(m.W_out.data, dtype=np.float32)

    ratios.append(("matmul/real", matmul_case(np.maximum(real_pre, 0), W_out, "real MLP")))
    ratios.append(("logsoftmax/real", logsoftmax_case(real_logits, "real logits")))
    ratios.append(("logsoftmax/large",
                   logsoftmax_case((real_logits * 20).astype(np.float32), "logits x20")))

    print("\n--- verdict against pre-registered criteria A1-A3 ---")
    for tag, r in ratios:
        state = "candidate SUPPORTED" if r <= 0.5 else (
            "REVERSED" if r >= 2.0 else "equivalent (A3)")
        print(f"  {tag:<20} ratio {r:7.3f}   {state}")
    equiv = all(0.5 < r < 2.0 for _, r in ratios)
    print()
    if equiv:
        print("A3 FIRES: numpy and torch are numerically equivalent at float32 on every")
        print("case measured. Candidates (a) and (b) are BOTH DEAD. O21 survives with no")
        print("candidate mechanism -- a reportable state, and a Limitations line.")
    else:
        print("A3 does NOT fire: at least one case shows a systematic accuracy difference.")
        print("See the per-case verdicts above for which candidate it supports.")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
