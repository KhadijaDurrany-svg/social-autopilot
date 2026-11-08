from __future__ import annotations

from slotwise.baselines import fixed_time, greedy, random_schedule
from slotwise.generate import generate_scenario


def _expected_count(scenario, ch):
    return min(ch.posts, scenario.capacity(ch))


def test_all_baselines_feasible():
    scenario = generate_scenario(11)
    for sched in (greedy(scenario), fixed_time(scenario), random_schedule(scenario, 11)):
        sched.validate_against(scenario)


def test_baselines_place_full_target_count():
    scenario = generate_scenario(12)
    for strat in (greedy(scenario), fixed_time(scenario), random_schedule(scenario, 12)):
        for ch in scenario.channels:
            assert strat.count_for(ch.name) == _expected_count(scenario, ch)


def test_random_is_deterministic_given_seed():
    scenario = generate_scenario(5)
    a = random_schedule(scenario, 123)
    b = random_schedule(scenario, 123)
    assert a.placements == b.placements


def test_random_differs_across_seeds():
    scenario = generate_scenario(5)
    a = random_schedule(scenario, 1)
    b = random_schedule(scenario, 2)
    assert a.placements != b.placements


def test_fixed_time_uses_consistent_slot_of_day():
    scenario = generate_scenario(6)
    sched = fixed_time(scenario, hour=9)
    ch = scenario.channels[0]
    if ch.daily_cap == 1:
        hours = {p.slot % scenario.slots_per_day for p in sched.placements if p.channel == ch.name}
        assert hours == {9 % scenario.slots_per_day}


def test_greedy_prefers_high_activity_slot():
    scenario = generate_scenario(8)
    ch = scenario.channels[0]
    sched = greedy(scenario)
    used = sched.slots_for(ch.name)
    # The single global maximum activity slot should be chosen by greedy.
    best_slot = max(range(scenario.horizon), key=lambda s: ch.activity[s])
    assert best_slot in used


def test_greedy_strategy_label():
    scenario = generate_scenario(9)
    assert greedy(scenario).strategy == "greedy"
    assert fixed_time(scenario).strategy == "fixed_time"
    assert random_schedule(scenario, 9).strategy == "random"
