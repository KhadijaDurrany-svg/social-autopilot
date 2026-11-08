"""The simulated-engagement oracle.

The oracle is the *ground truth* a schedule is judged against. The planner and all
baselines only ever see the scenario's *expected* activity; the oracle draws a
mean-preserving multiplicative noise term for every (channel, slot) and realizes
engagement as::

    realized[c][t] = expected[c][t] * lognormal_noise(c, t)      E[noise] == 1

A schedule's oracle score applies the same fatigue model to the realized activity. The
noise is seeded, so the oracle is deterministic given a seed, yet it breaks the tie
between "best in expectation" and "best in hindsight": the optimizer maximizes expected
value and therefore does *not* in general achieve the realized optimum, which keeps the
benchmark honest (no strategy is perfect).

``oracle_best`` computes the perfect-foresight upper bound by running the exact optimizer
on the realized activity. It is used to report each strategy's efficiency as a fraction
of the achievable maximum.
"""

from __future__ import annotations

import numpy as np

from slotwise.models import Scenario, Schedule
from slotwise.optimize import optimize_channel, schedule_value


class Oracle:
    """Deterministic simulated-engagement environment for one scenario."""

    def __init__(self, scenario: Scenario, seed: int, sigma: float = 0.45) -> None:
        if sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        self.scenario = scenario
        self.seed = seed
        self.sigma = sigma
        self._realized = self._draw_realized()

    def _draw_realized(self) -> dict[str, list[float]]:
        rng = np.random.default_rng((self.seed, 0xA17E))
        realized: dict[str, list[float]] = {}
        sigma = self.sigma
        for ch in self.scenario.channels:
            exp = np.asarray(ch.activity, dtype=float)
            if sigma == 0.0:
                noise = np.ones_like(exp)
            else:
                z = rng.normal(0.0, 1.0, size=exp.shape)
                # Mean-preserving lognormal: E[exp(sigma*z - sigma^2/2)] == 1.
                noise = np.exp(sigma * z - 0.5 * sigma * sigma)
            realized[ch.name] = [float(x) for x in (exp * noise)]
        return realized

    @property
    def realized(self) -> dict[str, list[float]]:
        return self._realized

    def score(self, schedule: Schedule) -> float:
        """Realized engagement of ``schedule`` under this oracle's drawn activity."""
        return schedule_value(self.scenario, schedule, activity=self._realized)

    def best(self) -> float:
        """Perfect-foresight upper bound: the exact optimum under realized activity."""
        total = 0.0
        for ch in self.scenario.channels:
            _, val = optimize_channel(
                self._realized[ch.name],
                ch.posts,
                ch.daily_cap,
                self.scenario.slots_per_day,
                self.scenario.fatigue,
            )
            total += val
        return total


__all__ = ["Oracle"]
