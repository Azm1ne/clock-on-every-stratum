"""Classify a training run from its saved history. One implementation, not five.

Two rules, both earned, and this module exists because each was independently
re-implemented and independently got wrong somewhere in the tree:

1. **Read test accuracy BY COLUMN NAME.** `hist[-1][3]` is test accuracy in the Kaggle
   kernels (`step, train_loss, test_loss, test_acc`) and TRAIN accuracy on the engine
   (`step, train_loss, test_loss, TRAIN_acc, test_acc`). The same index silently means a
   different quantity in the two arms.

2. **A run's endpoint is a WINDOW, never its last row.** C27 ("a network can de-grok") was
   `k06_horizon/WE_B_thesis_n119_s0` read at its final logged sample, 0.6798. It is a
   transient: 19 post-grok excursions below 0.90 occur across the 11 grokked runs, every one
   exactly one 200-step logging interval long, and that run's window median is 0.9987. Scored
   over 162 runs, the two rules disagree on 4 -- and the worst of them puts a *grokked* model
   into a control arm, which is the direction nobody checks.

Three states, never two. A binary gate turns every near-miss into whichever side it falls
on: k03's `n125_s1` at 0.9779 was being scored as the ungrokked negative control and quietly
destroying the contrast the control exists to provide.

Self-check: `PYTHONPATH=. .venv/bin/python src/analysis/runs.py`
"""
import numpy as np

GROK_ACC = 0.99   # above this: grokked, a data point
FAIL_ACC = 0.90   # below this: FAILED, a control
WINDOW = 10       # samples averaged at the endpoint


def test_acc_series(z):
    """The test-accuracy column of a saved history, by name. Raises if it is absent."""
    cols = [str(c) for c in z["hist_cols"]]
    assert "test_acc" in cols, f"hist_cols has no test_acc: {cols}"
    return np.asarray(z["hist"], dtype=float)[:, cols.index("test_acc")]


def final_acc(z, window=WINDOW):
    """The run's endpoint: the MEDIAN of its final `window` samples. Never the last row."""
    return float(np.median(test_acc_series(z)[-window:]))


def last_test_acc(z):
    """The FINAL ROW's test accuracy -- the checkpoint that `logits_all` was saved at.

    This is the right reading for a *consistency check* against saved logits, and the wrong
    one for *classifying* a run (use `state`). Keeping them as one function is how the
    last-row defect survived in `analyze_gate2.py`: its `check_split` compares a recomputed
    held-out accuracy against this value with a 0.01 tolerance, so routing the classifier to
    a window median would silently have made post-grok jitter look like "split does not
    reproduce" and skipped good runs. Two questions, two names.
    """
    return float(test_acc_series(z)[-1])


def state(z, window=WINDOW):
    """grokked / near-grok / FAILED."""
    a = final_acc(z, window)
    return ("grokked" if a > GROK_ACC else "FAILED" if a < FAIL_ACC else "near-grok"), a


def _selfcheck():
    class Z(dict):
        @property
        def files(self): return list(self)

    def mk(accs, cols=("step", "train_loss", "test_loss", "test_acc")):
        h = np.zeros((len(accs), len(cols)))
        h[:, cols.index("test_acc")] = accs
        return Z(hist=h, hist_cols=np.array(cols))

    # 1. The C27 case, which is the whole reason this module exists: a grokked run whose
    #    LAST SAMPLE is a one-interval excursion. The window says grokked; the last row says
    #    FAILED, and that is the direction that puts a grokked model into a control arm.
    z = mk([0.999] * 14 + [0.6798])
    assert state(z)[0] == "grokked", state(z)
    assert float(test_acc_series(z)[-1]) < FAIL_ACC, "the planted transient must be below the gate"

    # 2. The near-grok must be neither, from the window -- k03's n125_s1 reads 0.976 there.
    assert state(mk([0.976] * 12))[0] == "near-grok"
    assert state(mk([0.9989] * 12))[0] == "grokked"
    assert state(mk([0.6082] * 12))[0] == "FAILED"

    # 3. Column name, not index 3. On the engine's layout index 3 is TRAIN accuracy, so a
    #    run at train ~1.0 and test 0.6082 must read FAILED, not grokked.
    eng = ("step", "train_loss", "test_loss", "train_acc", "test_acc")
    h = np.zeros((12, 5)); h[:, 3] = 1.0; h[:, 4] = 0.6082
    ze = Z(hist=h, hist_cols=np.array(eng))
    assert state(ze)[0] == "FAILED", state(ze)
    assert abs(final_acc(ze) - 0.6082) < 1e-12

    # 4. A history shorter than the window is averaged over what it has, not padded.
    assert abs(final_acc(mk([0.5, 0.7, 0.9])) - 0.7) < 1e-12

    # 5. last_test_acc and state answer DIFFERENT questions on the same history, and the
    #    C27 run is exactly where they diverge. If these ever agree, one of them is wrong.
    z27 = mk([0.999] * 14 + [0.6798])
    assert abs(last_test_acc(z27) - 0.6798) < 1e-12
    assert state(z27)[0] == "grokked"

    print("runs.py: SELFCHECK PASS")


if __name__ == "__main__":
    _selfcheck()
