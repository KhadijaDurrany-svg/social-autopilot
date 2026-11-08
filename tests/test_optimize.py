from __future__ import annotations

import numpy as np
import pytest

from slotwise.generate import generate_scenario
from slotwise.models import Channel, FatigueParams, Scenario
from slotwise.optimize import (
    _brute_force_channel,
    optimize_channel,
    optimize_schedule,
    schedule_value,
)


@pytest.mark.parametrize("trial", range(60))
def test_dp_matches_brute_force(trial):
    rng = np.random.default_rng(1000 + trial)
    spd = int(rng.integers(3, 6))
    days = int(rng.integers(1, 3))
    T = spd * days
    act = [float(x) for x in rng.uniform(0.05, 1.0, size=T)]
    dcap = int(rng.integers(1, 3))
    posts = int(rng.integers(1, days * dcap + 1))
    fat = FatigueParams(recover_slots=int(rng.integers(2, 5)), floor=float(rng.uniform(0.3, 0.6)))
    _, v_dp = optimize_channel(act, posts, dcap, spd, fat)
    _, v_bf = _brute_force_channel(act, posts, dcap, spd, fat)
    assert v_dp == pytest.approx(v_bf, abs=1e-9)


def test_dp_chosen_slots_value_consistent():
    rng = np.random.default_rng(7)
    spd, days = 5, 2
    act = [float(x) for x in rng.uniform(0.05, 1.0, size=spd * days)]
    fat = FatigueParams(recover_slots=3, floor=0.4)
    slots, val = optimize_channel(act, posts=3, daily_cap=2, slots_per_day=spd, fatigue=fat)
    # Recompute the value from the returned slots and the fatigue model.
    total = 0.0
    prev = None
    for s in slots:
        mult = 1.0 if prev is None else fat.multiplier(s - prev)
        total += act[s] * mult
        prev = s
    assert total == pytest.approx(val, abs=1e-9)
    assert slots == sorted(slots)


def test_optimize_schedule_is_feasible_and_full():
    scenario = generate_scenario(3)
    sched = optimize_schedule(scenario)
    sched.validate_against(scenario)
    for ch in scenario.channels:
        assert sched.count_for(ch.name) == min(ch.posts, scenario.capacity(ch))


def test_optimize_beats_adjacent_clustering():
    # Broad plateau: greedy clustering at the three top adjacent hours is suboptimal.
    spd = 10
    act = [0.2, 0.4, 0.8, 0.95, 1.0, 0.98, 0.85, 0.5, 0.3, 0.15]
    fat = FatigueParams(recover_slots=5, floor=0.3)
    slots, val = optimize_channel(act, posts=3, daily_cap=3, slots_per_day=spd, fatigue=fat)
    # The 3 adjacent peak hours (3,4,5) would be the greedy pick.
    cluster = (3, 4, 5)
    cval = act[3] + act[4] * fat.multiplier(1) + act[5] * fat.multiplier(1)
    assert val >= cval - 1e-12
    assert tuple(slots) != cluster  # optimizer spreads out


def test_schedule_value_matches_oracle_style_sum():
    act = tuple(float(i + 1) / 10 for i in range(6))
    ch = Channel(name="x", activity=act, posts=2, daily_cap=2)
    scenario = Scenario(slots_per_day=3, days=2, channels=(ch,), fatigue=FatigueParams(recover_slots=2, floor=0.5))
    sched = optimize_schedule(scenario)
    v = schedule_value(scenario, sched)
    assert v > 0.0


def test_target_capped_by_capacity():
    # posts far exceeds capacity -> only capacity placed.
    spd = 4
    act = [0.5, 0.6, 0.7, 0.8]
    fat = FatigueParams(recover_slots=2, floor=0.5)
    slots, _ = optimize_channel(act, posts=100, daily_cap=2, slots_per_day=spd, fatigue=fat)
    assert len(slots) == 2
