"""Baseline slot-selection strategies to compare the optimizer against.

All baselines place the same number of posts per channel as the optimizer
(``min(posts, capacity)``) so comparisons are apples-to-apples: the strategies differ
only in *where* posts land, never in *how many*.

* ``fixed_time`` -- the common "post at the same hour every day" heuristic.
* ``random_schedule`` -- uniformly random feasible slots (a floor, not a strawman peak).
* ``greedy`` -- repeatedly take the highest expected-activity feasible slot. This is the
  natural myopic heuristic; it ignores fatigue, so it tends to cluster posts around the
  activity peak and pay a fatigue penalty.
"""

from __future__ import annotations

import numpy as np

from slotwise.models import Placement, Scenario, Schedule


def _target(scenario: Scenario, channel) -> int:  # noqa: ANN001 - internal helper
    return min(channel.posts, scenario.capacity(channel))


def fixed_time(scenario: Scenario, hour: int | None = None) -> Schedule:
    """Place posts at a fixed slot-of-day, filling earliest days first, respecting caps.

    If ``hour`` is ``None`` the mid-day slot is used (a reasonable, channel-agnostic
    default that does not peek at the activity curve).
    """
    placements: list[Placement] = []
    base = scenario.slots_per_day // 2 if hour is None else hour % scenario.slots_per_day
    for ch in scenario.channels:
        target = _target(scenario, ch)
        placed = 0
        # Offsets spread multiple same-day posts a few slots apart around the base hour.
        offsets = _spread_offsets(ch.daily_cap, scenario.slots_per_day)
        for day in range(scenario.days):
            if placed >= target:
                break
            for k in range(ch.daily_cap):
                if placed >= target:
                    break
                slot = day * scenario.slots_per_day + (base + offsets[k]) % scenario.slots_per_day
                placements.append(Placement(channel=ch.name, slot=slot))
                placed += 1
    return Schedule(placements=tuple(placements), strategy="fixed_time")


def _spread_offsets(daily_cap: int, slots_per_day: int) -> list[int]:
    """Deterministic offsets around the base hour for up to ``daily_cap`` same-day posts."""
    if daily_cap <= 1:
        return [0]
    step = max(1, slots_per_day // (daily_cap + 1))
    return [(-(daily_cap // 2) + i) * step for i in range(daily_cap)]


def random_schedule(scenario: Scenario, seed: int) -> Schedule:
    """Place posts at uniformly random feasible slots, respecting per-day caps."""
    rng = np.random.default_rng(seed)
    placements: list[Placement] = []
    for ch in scenario.channels:
        target = _target(scenario, ch)
        # Choose days (with per-day capacity) then a distinct slot-of-day within each.
        day_slots = {d: scenario.slots_per_day for d in range(scenario.days)}
        chosen: list[int] = []
        # Build a pool of (day) with remaining capacity and sample without replacement overall.
        remaining = target
        day_counts = {d: 0 for d in range(scenario.days)}
        used: set[int] = set()
        guard = 0
        while remaining > 0 and guard < 100000:
            guard += 1
            d = int(rng.integers(0, scenario.days))
            if day_counts[d] >= ch.daily_cap:
                continue
            h = int(rng.integers(0, scenario.slots_per_day))
            slot = d * scenario.slots_per_day + h
            if slot in used:
                continue
            used.add(slot)
            day_counts[d] += 1
            chosen.append(slot)
            remaining -= 1
        _ = day_slots  # retained for clarity; capacity is enforced via day_counts
        placements.extend(Placement(channel=ch.name, slot=s) for s in sorted(chosen))
    return Schedule(placements=tuple(placements), strategy="random")


def greedy(scenario: Scenario) -> Schedule:
    """Myopically take the highest expected-activity feasible slots (fatigue-blind)."""
    placements: list[Placement] = []
    for ch in scenario.channels:
        target = _target(scenario, ch)
        order = sorted(range(scenario.horizon), key=lambda s: ch.activity[s], reverse=True)
        day_counts = {d: 0 for d in range(scenario.days)}
        chosen: list[int] = []
        for slot in order:
            if len(chosen) >= target:
                break
            d = scenario.day_of(slot)
            if day_counts[d] >= ch.daily_cap:
                continue
            day_counts[d] += 1
            chosen.append(slot)
        placements.extend(Placement(channel=ch.name, slot=s) for s in sorted(chosen))
    return Schedule(placements=tuple(placements), strategy="greedy")


__all__ = ["fixed_time", "random_schedule", "greedy"]
