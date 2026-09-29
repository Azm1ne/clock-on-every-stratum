"""Lock the 2D DFT bin convention. Getting this backwards inverts every conclusion drawn
from the neuron term decomposition, and reasoning about it is not reliable -- the first
version of neuron_term_decomposition had (a+b) and (a-b) swapped.
"""
import numpy as np, tempfile, json
from pathlib import Path
from src.viz.mechinterp import neuron_term_decomposition

N, K = 113, 7


def synth(kind):
    a = np.arange(N)[:, None]; b = np.arange(N)[None, :]
    f = {"a+b": np.cos(2 * np.pi * K * (a + b) / N),
         "a-b": np.cos(2 * np.pi * K * (a - b) / N),
         "a only": np.cos(2 * np.pi * K * a / N) * np.ones_like(b),
         "b only": np.cos(2 * np.pi * K * b / N) * np.ones_like(a)}[kind]
    return np.repeat(f.reshape(N * N, 1), 4, axis=1).astype(np.float32)


def main():
    print("2D DFT bin convention -- each planted term must be attributed to itself:")
    with tempfile.TemporaryDirectory() as d:
        for kind in ["a+b", "a-b", "a only", "b only"]:
            f = Path(d) / "x.npz"
            np.savez(f, mlp_acts=synth(kind))
            frac = neuron_term_decomposition(f, N, verbose=False)
            got = max(frac, key=lambda k: frac[k].mean())
            share = frac[kind].mean()
            print(f"  planted {kind:<7} -> attributed {got:<7} "
                  f"({share:.1%} of energy)   {'OK' if got == kind else 'FAIL'}")
            assert got == kind, f"planted {kind}, attributed {got}"
            assert share > 0.95, f"{kind}: only {share:.1%} attributed"
    print("\ntest_mechinterp: PASS")


if __name__ == "__main__":
    main()
