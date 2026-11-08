"""The slotwise optimizer.

The objective is additive over placements. For a placement on channel ``c`` at slot
``t`` whose previous same-channel placement was ``gap`` slots earlier::

    value(t, gap) = activity[c][t] * fatigue.multiplier(gap)

The first placement on a channel has no predecessor and is unpenalized. Posts on
different channels do not interact, so the problem decomposes per channel. Within a
channel we must choose which slots to use (one post each) to maximize total value,
subject to the per-day cap and the number of posts to place.

Because fatigue only looks back to the immediately preceding placement, each channel's
subproblem is solved *exactly* by a left-to-right dynamic program whose state is
``(posts_placed, posts_today, gap_since_last)``. ``gap_since_last`` is clamped at the
fatigue recovery horizon because any larger gap behaves identically (multiplier 1.0).

This module optimizes the *expected* activity supplied in the scenario. The same solver
is reused by the oracle to compute a perfect-foresight upper bound (see
:mod:`slotwise.oracle`) by passing realized activity in place of expected activity.
"""

from __future__ import annotations

from dataclasses import dataclass

from slotwise.models import Channel, FatigueParams, Placement, Scenario, Schedule

# Sentinel gap meaning "no previous placement" -> multiplier 1.0. Clamped to recover_slots.
_NO_PREV = -1


@dataclass(frozen=True)
class _State:
    placed: int
    today: int
    gap: int  # clamped gap since last placement; _NO_PREV if none yet


def _clamp_gap(gap: int, recover: int) -> int:
    if gap == _NO_PREV:
        return _NO_PREV
    return gap if gap < recover else recover


def optimize_channel(
    activity: list[float] | tuple[float, ...],
    posts: int,
    daily_cap: int,
    slots_per_day: int,
    fatigue: FatigueParams,
) -> tuple[list[int], float]:
    """Return ``(chosen_slots, total_value)`` for one channel, exactly maximizing value.

    ``chosen_slots`` is sorted ascending and has length ``min(posts, capacity)``.
    """
    horizon = len(activity)
    days = horizon // slots_per_day if horizon % slots_per_day == 0 else horizon // slots_per_day + 1
    target = min(posts, days * daily_cap)
    recover = fatigue.recover_slots

    # dp maps a state -> (best_value, backpointer). We iterate slot by slot.
    # best[state] = best value of decisions over slots [0..t-1] arriving at slot t.
    NEG = float("-inf")
    # Represent states as tuples for dict keys.
    best: dict[tuple[int, int, int], float] = {(0, 0, _NO_PREV): 0.0}
    # parent[t][state] = (prev_state, placed_here_bool)
    parents: list[dict[tuple[int, int, int], tuple[tuple[int, int, int], bool]]] = []

    for t in range(horizon):
        day = t // slots_per_day
        next_day = (t + 1) // slots_per_day
        day_rolls = next_day != day
        nxt: dict[tuple[int, int, int], float] = {}
        par: dict[tuple[int, int, int], tuple[tuple[int, int, int], bool]] = {}

        for (placed, today, gap), val in best.items():
            # ``gap`` is the number of slots elapsed since the last placement, measured
            # as of slot ``t`` (or _NO_PREV if no post has been placed on this channel).

            # Option 1: skip slot t. One more slot elapses before slot t+1.
            new_gap = _NO_PREV if gap == _NO_PREV else _clamp_gap(gap + 1, recover)
            new_today = 0 if day_rolls else today
            key = (placed, new_today, new_gap)
            if val > nxt.get(key, NEG):
                nxt[key] = val
                par[key] = ((placed, today, gap), False)

            # Option 2: place a post at slot t (if allowed).
            if placed < target and today < daily_cap:
                mult = 1.0 if gap == _NO_PREV else fatigue.multiplier(gap)
                gain = activity[t] * mult
                # After placing at t, the last placement is now t; as of slot t+1 exactly
                # one slot has elapsed, so the arriving gap is 1.
                placed_today = today + 1
                new_today2 = 0 if day_rolls else placed_today
                new_gap2 = _clamp_gap(1, recover)
                key2 = (placed + 1, new_today2, new_gap2)
                nv = val + gain
                if nv > nxt.get(key2, NEG):
                    nxt[key2] = nv
                    par[key2] = ((placed, today, gap), True)

        best = nxt
        parents.append(par)

    # Find the best terminal state with exactly `target` posts placed.
    best_key = None
    best_val = NEG
    for key, val in best.items():
        if key[0] == target and val > best_val:
            best_val = val
            best_key = key
    if best_key is None:
        # target may exceed what fits (shouldn't happen given target<=capacity); fall back.
        for key, val in best.items():
            if val > best_val:
                best_val = val
                best_key = key

    # Backtrack to recover chosen slots.
    chosen: list[int] = []
    state = best_key
    for t in range(horizon - 1, -1, -1):
        prev_state, placed_here = parents[t][state]
        if placed_here:
            chosen.append(t)
        state = prev_state
    chosen.reverse()
    return chosen, (0.0 if best_val == NEG else best_val)


def optimize_schedule(scenario: Scenario, strategy: str = "slotwise") -> Schedule:
    """Return the value-maximizing :class:`Schedule` for ``scenario`` (expected activity)."""
    placements: list[Placement] = []
    for ch in scenario.channels:
        slots, _ = optimize_channel(
            ch.activity, ch.posts, ch.daily_cap, scenario.slots_per_day, scenario.fatigue
        )
        placements.extend(Placement(channel=ch.name, slot=s) for s in slots)
    return Schedule(placements=tuple(placements), strategy=strategy)


def schedule_value(scenario: Scenario, schedule: Schedule, activity: dict[str, list[float]] | None = None) -> float:
    """Compute the additive objective of ``schedule`` under ``activity``.

    If ``activity`` is ``None`` the scenario's expected activity is used. This is the
    canonical scorer used by both the optimizer's objective and the oracle.
    """
    total = 0.0
    fatigue = scenario.fatigue
    for ch in scenario.channels:
        curve = activity[ch.name] if activity is not None else list(ch.activity)
        slots = schedule.slots_for(ch.name)
        prev: int | None = None
        for s in slots:
            if prev is None:
                mult = 1.0
            else:
                mult = fatigue.multiplier(s - prev)
            total += curve[s] * mult
            prev = s
    return total


def _brute_force_channel(
    activity: list[float],
    posts: int,
    daily_cap: int,
    slots_per_day: int,
    fatigue: FatigueParams,
) -> tuple[list[int], float]:
    """Reference solver used only by the test-suite to certify the DP. Exponential time."""
    from itertools import combinations

    horizon = len(activity)
    days = horizon // slots_per_day if horizon % slots_per_day == 0 else horizon // slots_per_day + 1
    target = min(posts, days * daily_cap)

    def value_of(slots: tuple[int, ...]) -> float:
        total = 0.0
        prev: int | None = None
        for s in slots:
            mult = 1.0 if prev is None else fatigue.multiplier(s - prev)
            total += activity[s] * mult
            prev = s
        return total

    best_slots: tuple[int, ...] = ()
    best_val = float("-inf")
    for combo in combinations(range(horizon), target):
        per_day: dict[int, int] = {}
        ok = True
        for s in combo:
            d = s // slots_per_day
            per_day[d] = per_day.get(d, 0) + 1
            if per_day[d] > daily_cap:
                ok = False
                break
        if not ok:
            continue
        v = value_of(combo)
        if v > best_val:
            best_val = v
            best_slots = combo
    return list(best_slots), (0.0 if best_val == float("-inf") else best_val)


__all__ = ["optimize_channel", "optimize_schedule", "schedule_value", "Channel"]
