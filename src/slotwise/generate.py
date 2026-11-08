"""Seeded synthetic scenario generator.

Scenarios are fully determined by an integer seed, so benchmarks are reproducible. The
generator is deliberately tuned to be *non-trivial*: activity curves have a broad
mid-day plateau, which tempts the myopic greedy baseline into clustering posts into
adjacent high-activity hours and paying a fatigue penalty. The optimizer wins by trading
a little raw activity for better spacing -- but only in expectation, because the oracle
adds mean-preserving multiplicative noise (see :mod:`slotwise.oracle`).
"""

from __future__ import annotations

import numpy as np

from slotwise.models import Channel, FatigueParams, Scenario

_CHANNEL_POOL = ("x", "instagram", "linkedin", "tiktok", "facebook")


def _daily_shape(rng: np.random.Generator, slots_per_day: int) -> np.ndarray:
    """A broad, single-plateau day curve with a channel-specific peak hour and width."""
    peak = rng.uniform(0.35, 0.65) * slots_per_day  # mid-day-ish
    width = rng.uniform(0.14, 0.22) * slots_per_day  # broad -> many near-equal good hours
    hours = np.arange(slots_per_day, dtype=float)
    hump = np.exp(-0.5 * ((hours - peak) / width) ** 2)
    base = 0.08  # small floor so off-peak hours are not literally zero
    shape = base + (1.0 - base) * hump
    return shape


def generate_scenario(seed: int, days: int = 7, slots_per_day: int = 24) -> Scenario:
    """Generate a reproducible :class:`Scenario` from ``seed``."""
    rng = np.random.default_rng(seed)
    n_channels = int(rng.integers(2, 5))  # 2..4 channels
    names = list(rng.permutation(np.array(_CHANNEL_POOL)))[:n_channels]

    # Day-of-week weighting: a smooth weekly modulation shared across channels but phased.
    dow = np.arange(days, dtype=float)

    channels: list[Channel] = []
    for name in names:
        shape = _daily_shape(rng, slots_per_day)
        scale = rng.uniform(0.6, 1.4)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        week = 0.75 + 0.25 * np.sin(2.0 * np.pi * dow / max(days, 1) + phase)
        curve = np.empty(days * slots_per_day, dtype=float)
        for d in range(days):
            curve[d * slots_per_day : (d + 1) * slots_per_day] = scale * week[d] * shape
        daily_cap = int(rng.integers(2, 4))  # 2 or 3 posts/day -> room to cluster
        capacity = days * daily_cap
        # Place enough posts that caps and spacing both matter, but all are placeable.
        posts = int(rng.integers(max(days, 1), capacity + 1))
        channels.append(
            Channel(
                name=str(name),
                activity=tuple(float(x) for x in curve),
                posts=posts,
                daily_cap=daily_cap,
            )
        )

    recover = int(rng.integers(4, 8))
    floor = float(rng.uniform(0.3, 0.5))
    fatigue = FatigueParams(recover_slots=recover, floor=floor)
    return Scenario(slots_per_day=slots_per_day, days=days, channels=tuple(channels), fatigue=fatigue)


__all__ = ["generate_scenario"]
