"""Sparsity metrics on an energy spectrum. Basis-independent by construction."""
import numpy as np


def gini(e):
    """0 = uniform, ->1 = all energy in one component.

    2606.17399 reports 0.58 (multiplicative) vs 0.07 (additive) for a*b mod 113.
    """
    e = np.sort(np.abs(np.asarray(e, dtype=float)))
    n = len(e)
    if n == 0 or e.sum() == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float((2 * (i * e).sum()) / (n * e.sum()) - (n + 1) / n)


def participation_ratio(e):
    """Effective number of active components: n for uniform, 1 for a spike."""
    e = np.abs(np.asarray(e, dtype=float))
    s2 = (e ** 2).sum()
    return float(e.sum() ** 2 / s2) if s2 > 0 else 0.0


def topk_energy(e, k):
    e = np.sort(np.abs(np.asarray(e, dtype=float)))[::-1]
    return float(e[:k].sum() / e.sum()) if e.sum() > 0 else 0.0


def n_key_frequencies(e, threshold=0.9):
    """Smallest number of components holding `threshold` of the total energy."""
    e = np.sort(np.abs(np.asarray(e, dtype=float)))[::-1]
    if e.sum() == 0:
        return 0
    return int(np.searchsorted(np.cumsum(e) / e.sum(), threshold) + 1)


def key_freqs_5x_median(norms):
    """2606.17399's detector: frequencies whose norm exceeds 5x the median norm.

    Their reported result for a*b mod 113: 4 key frequencies at {2, 8, 47, 56}.
    Kept separate from n_key_frequencies (a 90%-energy threshold) because the two
    disagree on borderline spectra and only this one is comparable to the paper.
    """
    x = np.abs(np.asarray(norms, dtype=float))
    med = np.median(x)
    return sorted(int(i) for i in np.where(x > 5 * med)[0])
