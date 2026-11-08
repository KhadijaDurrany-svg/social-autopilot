from __future__ import annotations

import pytest
from pydantic import ValidationError

from slotwise.models import (
    Approval,
    Channel,
    FatigueParams,
    Placement,
    Scenario,
    Schedule,
)


def _scenario() -> Scenario:
    act = tuple(float(i % 4) + 0.1 for i in range(8))
    ch = Channel(name="x", activity=act, posts=2, daily_cap=2)
    return Scenario(slots_per_day=4, days=2, channels=(ch,), fatigue=FatigueParams(recover_slots=3, floor=0.4))


def test_fatigue_multiplier_endpoints():
    f = FatigueParams(recover_slots=4, floor=0.5)
    assert f.multiplier(1) == pytest.approx(0.5 + 0.5 * 0.25)
    assert f.multiplier(4) == 1.0
    assert f.multiplier(10) == 1.0


def test_fatigue_multiplier_monotone():
    f = FatigueParams(recover_slots=6, floor=0.3)
    vals = [f.multiplier(g) for g in range(1, 7)]
    assert all(b >= a for a, b in zip(vals, vals[1:], strict=False))


def test_fatigue_rejects_gap_zero():
    f = FatigueParams(recover_slots=3, floor=0.4)
    with pytest.raises(ValueError):
        f.multiplier(0)


def test_fatigue_floor_bounds():
    with pytest.raises(ValidationError):
        FatigueParams(recover_slots=3, floor=0.0)
    with pytest.raises(ValidationError):
        FatigueParams(recover_slots=3, floor=1.0)
    with pytest.raises(ValidationError):
        FatigueParams(recover_slots=0, floor=0.5)


def test_channel_rejects_negative_activity():
    with pytest.raises(ValidationError):
        Channel(name="x", activity=(-1.0, 0.2), posts=1, daily_cap=1)


def test_scenario_horizon_and_day_of():
    s = _scenario()
    assert s.horizon == 8
    assert s.day_of(0) == 0
    assert s.day_of(4) == 1
    assert s.capacity(s.channels[0]) == 4


def test_scenario_rejects_activity_length_mismatch():
    ch = Channel(name="x", activity=(0.1, 0.2, 0.3), posts=1, daily_cap=1)
    with pytest.raises(ValidationError):
        Scenario(slots_per_day=4, days=2, channels=(ch,), fatigue=FatigueParams(recover_slots=3, floor=0.4))


def test_scenario_rejects_duplicate_channel_names():
    act = tuple(0.1 for _ in range(8))
    a = Channel(name="dup", activity=act, posts=1, daily_cap=1)
    b = Channel(name="dup", activity=act, posts=1, daily_cap=1)
    with pytest.raises(ValidationError):
        Scenario(slots_per_day=4, days=2, channels=(a, b), fatigue=FatigueParams(recover_slots=3, floor=0.4))


def test_schedule_validate_ok():
    s = _scenario()
    sched = Schedule(placements=(Placement(channel="x", slot=1), Placement(channel="x", slot=5)))
    sched.validate_against(s)  # no raise


def test_schedule_rejects_double_booking():
    s = _scenario()
    sched = Schedule(placements=(Placement(channel="x", slot=1), Placement(channel="x", slot=1)))
    with pytest.raises(ValueError):
        sched.validate_against(s)


def test_schedule_rejects_daily_cap_violation():
    s = _scenario()  # daily_cap=2, slots_per_day=4 -> day 0 is slots 0..3
    sched = Schedule(
        placements=(
            Placement(channel="x", slot=0),
            Placement(channel="x", slot=1),
            Placement(channel="x", slot=2),
        )
    )
    with pytest.raises(ValueError):
        sched.validate_against(s)


def test_schedule_rejects_out_of_range_and_unknown_channel():
    s = _scenario()
    with pytest.raises(ValueError):
        Schedule(placements=(Placement(channel="x", slot=99),)).validate_against(s)
    with pytest.raises(ValueError):
        Schedule(placements=(Placement(channel="nope", slot=0),)).validate_against(s)


def test_schedule_counts_and_slots():
    sched = Schedule(
        placements=(Placement(channel="x", slot=5), Placement(channel="x", slot=1)),
        strategy="t",
    )
    assert sched.count_for("x") == 2
    assert sched.slots_for("x") == [1, 5]


def test_approval_defaults_unconfirmed():
    a = Approval(approved_by="khadija")
    assert a.confirmed is False
