from __future__ import annotations

import numpy as np
import pytest

from slotwise.stats import bootstrap_mean_ci, paired_permutation_pvalue, win_rate


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(5.0, 1.0, size=500)
    mean, lo, hi = bootstrap_mean_ci(data, n_boot=2000, seed=1)
    assert lo <= mean <= hi
    assert mean == pytest.approx(data.mean())
    assert lo < hi


def test_bootstrap_single_value():
    mean, lo, hi = bootstrap_mean_ci(np.array([3.0]))
    assert mean == lo == hi == 3.0


def test_permutation_pvalue_large_effect_hits_floor():
    diffs = np.full(100, 0.5)
    p = paired_permutation_pvalue(diffs, n_perm=5000, seed=0)
    assert p == pytest.approx(1.0 / 5001.0)


def test_permutation_pvalue_zero_effect_is_one():
    diffs = np.zeros(50)
    assert paired_permutation_pvalue(diffs) == 1.0


def test_permutation_pvalue_noise_is_large():
    rng = np.random.default_rng(3)
    diffs = rng.normal(0.0, 1.0, size=200)  # no true effect
    p = paired_permutation_pvalue(diffs, n_perm=5000, seed=2)
    assert p > 0.05


def test_permutation_pvalue_is_positive_and_bounded():
    rng = np.random.default_rng(4)
    diffs = rng.normal(0.1, 1.0, size=80)
    p = paired_permutation_pvalue(diffs, n_perm=3000, seed=5)
    assert 0.0 < p <= 1.0


def test_win_rate():
    assert win_rate(np.array([1.0, -1.0, 2.0, 0.0])) == pytest.approx(0.5)


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        bootstrap_mean_ci(np.array([]))
    with pytest.raises(ValueError):
        paired_permutation_pvalue(np.array([]))
    with pytest.raises(ValueError):
        win_rate(np.array([]))
