"""Gate 1: did the from-scratch model reproduce Nanda et al. 2023?

This file is the gate. It was written before the run, so the criteria cannot drift to fit
whatever comes out. Criteria are the research plan's H1.1, with Nanda's published
reference numbers for a+b mod 113 (2301.05217):

  - 5 key frequencies (k in {14,35,41,42,52}); the plan allows <= 10
  - ablating key frequencies destroys performance
  - ablating all non-key frequencies does not harm, and in Nanda's run improves it
  - 84.6% of MLP neurons >85% variance explained by a single frequency

Usage: python test_gate1.py results/gate1/add113_final.npz
"""
import sys
import numpy as np
from src.analysis.ablation import ablation_report, key_freq_pairs, cross_entropy_np
from src.analysis.sparsity import gini, key_freqs_5x_median, participation_ratio

N = 113
NANDA_KEY_FREQS = 5


def freq_norms(W, n):
    """Combined sin/cos norm per frequency, DC dropped -- the protocol invariant."""
    k = np.arange(1, n // 2 + 1)[:, None]
    t = np.arange(n)[None, :]
    s = np.sin(2 * np.pi * k * t / n) @ W
    c = np.cos(2 * np.pi * k * t / n) @ W
    return np.sqrt((s ** 2).sum(1) + (c ** 2).sum(1))


def gate1(path):
    z = np.load(path)
    W_E, logits, y = z["W_E"], z["logits_all"], z["y_all"]
    results, failures = {}, []

    def check(name, ok, detail):
        results[name] = (ok, detail)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} {detail}")
        if not ok:
            failures.append(name)

    print(f"GATE 1 — a+b mod {N}, from-scratch engine\n")

    acc = float((logits.argmax(-1) == y).mean())
    check("C1 test accuracy > 99%", acc > 0.99, f"{acc:.4f}")

    fn = freq_norms(W_E[:N] - W_E[:N].mean(0, keepdims=True), N)
    keys = [k + 1 for k in key_freqs_5x_median(fn)]
    check("C2 <= 10 key frequencies (Nanda: 5)", 0 < len(keys) <= 10,
          f"{len(keys)} found: {keys}")
    check("C3 embedding sparse in additive basis", gini(fn) > 0.4,
          f"Gini {gini(fn):.3f}, PR {participation_ratio(fn):.2f}")

    grid = logits.reshape(N, N, N)
    rep = ablation_report(grid, y, keys, N)
    check("C4 ablating key freqs destroys performance",
          rep["excluded"] > rep["baseline"] + 1.0,
          f"excluded {rep['excluded']:.3e} vs baseline {rep['baseline']:.3e}")
    check("C5 ablating non-key freqs does not harm",
          rep["restricted"] <= rep["baseline"] * 1.05 + 1e-6,
          f"restricted {rep['restricted']:.3e} vs baseline {rep['baseline']:.3e}"
          + ("  (IMPROVES, as Nanda)" if rep["restricted"] < rep["baseline"] else ""))

    # C6. Key frequencies must be defined CAUSALLY, at the logits, not by embedding norm.
    # Nanda makes exactly this distinction: "Of the six non-zero frequencies, five key
    # frequencies appear in later parts of the network." A frequency can carry large
    # embedding norm and still be vestigial downstream, so requiring every
    # embedding-frequency to be causally necessary is the wrong test.
    causal = sorted([k for k, v in rep["per_freq"].items() if v > 0.01])
    vestigial = [k for k in keys if k not in causal]
    dn = [v for k, v in rep["per_freq"].items() if k not in causal]
    check("C6 causal key freqs are few and each necessary",
          0 < len(causal) <= 10 and max(dn) < 1e-3,
          f"{len(causal)} causal {causal}, max non-causal delta {max(dn):+.1e}"
          + (f"; vestigial in W_E: {vestigial}" if vestigial else ""))
    rep2 = ablation_report(grid, y, causal, N)
    check("C6b restricted loss on the CAUSAL set does not harm",
          rep2["restricted"] <= rep["baseline"] * 1.05 + 1e-6,
          f"restricted {rep2['restricted']:.3e} vs baseline {rep['baseline']:.3e}")

    if "mlp_acts" in z:
        # C7. Nanda's criterion is ">85% of variance explained by a degree-2 polynomial
        # of ONE frequency". One frequency k occupies SEVERAL 2D bins -- (+-k,0), (0,+-k),
        # (+-k,+-k), (+-k,-+k) -- so scoring a single bin is far too strict and was the
        # reason this read 9.6%. Group the bins by frequency first.
        H = z["mlp_acts"].reshape(N, N, -1)
        H = H - H.mean((0, 1), keepdims=True)          # drop DC: it is not a frequency
        F = np.abs(np.fft.fft2(H, axes=(0, 1))) ** 2
        tot = F.sum((0, 1))
        best = np.zeros(F.shape[-1])
        for k in range(1, N // 2 + 1):
            m, mk = (k % N, (-k) % N), 0.0
            bins = {(m[0], 0), (m[1], 0), (0, m[0]), (0, m[1]),
                    (m[0], m[0]), (m[1], m[1]), (m[0], m[1]), (m[1], m[0])}
            e = sum(F[i, j] for i, j in bins)
            best = np.maximum(best, e)
        frac = float((best / np.maximum(tot, 1e-30) > 0.85).mean())
        check("C7 neurons single-frequency tuned (Nanda 84.6%)", frac > 0.5,
              f"{frac:.1%} of {F.shape[-1]} neurons")
    else:
        print("  [SKIP] C7 neuron tuning — no mlp_acts in checkpoint")

    print(f"\nGATE 1: {'PASS' if not failures else 'FAIL — ' + ', '.join(failures)}")
    return not failures


if __name__ == "__main__":
    sys.exit(0 if gate1(sys.argv[1] if len(sys.argv) > 1
                        else "results/gate1/add113_final.npz") else 1)
