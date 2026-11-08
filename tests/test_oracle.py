from __future__ import annotations

import numpy as np
import pytest

from slotwise.baselines import greedy
from slotwise.generate import generate_scenario
from slotwise.optimize import optimize_schedule, schedule_value
from slotwise.oracle import Oracle


def test_oracle_deterministic_given_seed():
    scenario = generate_scenario(21)
    a = Oracle(scenario, seed=21)
    b = Oracle(scenario, seed=21)
    assert a.realized == b.realized


def test_oracle_sigma_zero_equals_expected():
    scenario = generate_scenario(22)
    oracle = Oracle(scenario, seed=22, sigma=0.0)
    for ch in scenario.channels:
        assert oracle.realized[ch.name] == pytest.approx(list(ch.activity))
    # With no noise, the oracle score equals the expected objective.
    sched = optimize_schedule(scenario)
    assert oracle.score(sched) == pytest.approx(schedule_value(scenario, sched))


def test_oracle_noise_is_mean_preserving():
    scenario = generate_scenario(23)
    oracle = Oracle(scenario, seed=23, sigma=0.5)
    ch = scenario.channels[0]
    exp = np.asarray(ch.activity)
    real = np.asarray(oracle.realized[ch.name])
    ratio = real[exp > 0] / exp[exp > 0]
    # Mean of the multiplicative noise should be close to 1 across the horizon.
    assert ratio.mean() == pytest.approx(1.0, abs=0.15)


def test_oracle_best_is_upper_bound():
    scenario = generate_scenario(24)
    oracle = Oracle(scenario, seed=24, sigma=0.4)
    best = oracle.best()
    for sched in (optimize_schedule(scenario), greedy(scenario)):
        assert oracle.score(sched) <= best + 1e-9


def test_oracle_sigma_zero_optimizer_is_optimal():
    # Without noise, the foresight-free optimizer equals the perfect-foresight optimum.
    scenario = generate_scenario(25)
    oracle = Oracle(scenario, seed=25, sigma=0.0)
    sched = optimize_schedule(scenario)
    assert oracle.score(sched) == pytest.approx(oracle.best(), abs=1e-9)


def test_oracle_rejects_negative_sigma():
    scenario = generate_scenario(26)
    with pytest.raises(ValueError):
        Oracle(scenario, seed=26, sigma=-0.1)
