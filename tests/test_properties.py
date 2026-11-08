"""Property-based tests (Hypothesis).

These certify invariants over wide input ranges: the DP is always optimal (vs brute
force), schedules are always feasible, and the fatigue multiplier is well-behaved.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from slotwise.baselines import fixed_time, greedy, random_schedule
from slotwise.generate import generate_scenario
from slotwise.models import FatigueParams
from slotwise.optimize import _brute_force_channel, optimize_channel, optimize_schedule

_SETTINGS = settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    slots_per_day=st.integers(min_value=3, max_value=5),
    days=st.integers(min_value=1, max_value=2),
    daily_cap=st.integers(min_value=1, max_value=2),
    posts=st.integers(min_value=1, max_value=6),
    recover=st.integers(min_value=2, max_value=5),
    floor=st.floats(min_value=0.3, max_value=0.6),
)
def test_dp_is_optimal_property(seed, slots_per_day, days, daily_cap, posts, recover, floor):
    rng = np.random.default_rng(seed)
    act = [float(x) for x in rng.uniform(0.05, 1.0, size=slots_per_day * days)]
    fat = FatigueParams(recover_slots=recover, floor=floor)
    _, v_dp = optimize_channel(act, posts, daily_cap, slots_per_day, fat)
    _, v_bf = _brute_force_channel(act, posts, daily_cap, slots_per_day, fat)
    assert abs(v_dp - v_bf) < 1e-9


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_all_strategies_feasible_property(seed):
    scenario = generate_scenario(seed)
    for sched in (
        optimize_schedule(scenario),
        greedy(scenario),
        fixed_time(scenario),
        random_schedule(scenario, seed),
    ):
        sched.validate_against(scenario)


@settings(max_examples=100, deadline=None)
@given(
    recover=st.integers(min_value=2, max_value=12),
    floor=st.floats(min_value=0.05, max_value=0.95),
    gap=st.integers(min_value=1, max_value=20),
)
def test_fatigue_multiplier_bounds_property(recover, floor, gap):
    f = FatigueParams(recover_slots=recover, floor=floor)
    m = f.multiplier(gap)
    assert floor <= m <= 1.0 + 1e-12
    if gap >= recover:
        assert m == 1.0
