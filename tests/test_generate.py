from __future__ import annotations

from slotwise.generate import generate_scenario


def test_generate_is_deterministic():
    a = generate_scenario(42)
    b = generate_scenario(42)
    assert a == b


def test_generate_varies_with_seed():
    a = generate_scenario(1)
    b = generate_scenario(2)
    assert a != b


def test_generated_scenario_is_valid_and_placeable():
    for seed in range(20):
        s = generate_scenario(seed)
        assert s.horizon == s.days * s.slots_per_day
        assert 2 <= len(s.channels) <= 4
        for ch in s.channels:
            assert len(ch.activity) == s.horizon
            # Posts should never exceed capacity, so every post is placeable.
            assert ch.posts <= s.capacity(ch)
            assert ch.posts >= 1


def test_generate_custom_horizon():
    s = generate_scenario(3, days=3, slots_per_day=12)
    assert s.days == 3
    assert s.slots_per_day == 12
    assert s.horizon == 36
