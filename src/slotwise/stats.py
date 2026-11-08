"""Small, dependency-light statistics used by the benchmark.

All routines are deterministic given a seed and operate on plain numpy arrays. They are
unit-tested against known cases (e.g. a permutation test on an all-positive paired
difference yields the minimal attainable p-value).
"""

from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray, confidence: float = 0.95, n_boot: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Return ``(mean, lo, hi)`` -- the sample mean and a percentile bootstrap CI."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        raise ValueError("values must be non-empty")
    rng = np.random.default_rng(seed)
    mean = float(values.mean())
    if n == 1:
        return mean, mean, mean
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return mean, lo, hi


def paired_permutation_pvalue(
    diffs: np.ndarray, n_perm: int = 20000, seed: int = 0
) -> float:
    """Two-sided paired permutation (sign-flip) p-value for ``mean(diffs) == 0``.

    Uses random sign flips of the paired differences. The returned p-value uses the
    standard ``(count + 1) / (n_perm + 1)`` estimator, so it is strictly positive and can
    never be reported as exactly zero.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    if n == 0:
        raise ValueError("diffs must be non-empty")
    observed = abs(float(diffs.mean()))
    if np.allclose(diffs, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    count = int(np.count_nonzero(perm_means >= observed - 1e-12))
    return (count + 1) / (n_perm + 1)


def win_rate(diffs: np.ndarray) -> float:
    """Fraction of paired differences strictly greater than zero."""
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size == 0:
        raise ValueError("diffs must be non-empty")
    return float(np.count_nonzero(diffs > 0.0) / diffs.size)


__all__ = ["bootstrap_mean_ci", "paired_permutation_pvalue", "win_rate"]
