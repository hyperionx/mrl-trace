"""Small statistics helpers for the figures and reported numbers.

The manuscript figures report bootstrapped 95% confidence intervals over the
random seeds (the comp-neuro convention) rather than mean +/- s.d. ``bootstrap_ci``
is the single primitive used everywhere; ``summarise`` packages a mean with its CI
for convenient reporting.
"""
from __future__ import annotations

import numpy as np

__all__ = ["bootstrap_ci", "summarise"]


def bootstrap_ci(values, n_boot=10000, ci=95, seed=0):
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Parameters
    ----------
    values : array-like
        Per-seed (or per-sample) scalar observations.
    n_boot : int
        Number of bootstrap resamples.
    ci : float
        Central confidence level in percent (95 -> 2.5/97.5 percentiles).
    seed : int
        RNG seed, so the interval is deterministic and reproducible.

    Returns
    -------
    (lo, hi) : tuple of float
        Lower and upper confidence bounds on the mean. For a single observation
        the bound collapses to that value.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return (float("nan"), float("nan"))
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    half = (100 - ci) / 2
    lo, hi = np.percentile(means, [half, 100 - half])
    return (float(lo), float(hi))


def summarise(values, n_boot=10000, ci=95, seed=0):
    """Return ``dict(mean, lo, hi, sd, n)`` for a per-seed observation array.

    ``lo``/``hi`` are the bootstrap CI bounds; ``sd`` is the sample standard
    deviation (kept for continuity with earlier mean +/- s.d. reporting).
    """
    v = np.asarray(values, dtype=float).ravel()
    lo, hi = bootstrap_ci(v, n_boot=n_boot, ci=ci, seed=seed)
    return {
        "mean": float(v.mean()) if v.size else float("nan"),
        "lo": lo,
        "hi": hi,
        "sd": float(v.std()) if v.size else float("nan"),
        "n": int(v.size),
    }
