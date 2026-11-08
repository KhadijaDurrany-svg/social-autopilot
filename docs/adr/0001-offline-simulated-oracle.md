# ADR 0001: An offline, simulated-engagement oracle

## Status

Accepted.

## Context

slotwise schedules social-media posts to maximize engagement. Real engagement can only
be observed after posting, and measuring it would require live platform APIs, credentials
and network access. That makes a tool non-reproducible, impossible to test in CI, and
hard to audit: anyone evaluating the scheduler would have to trust numbers that cannot be
regenerated.

## Decision

Engagement is defined by an explicit, deterministic **simulated oracle** rather than a
live platform. The oracle is seeded, so a given scenario always produces the same ground
truth, and it runs entirely offline. The planner and baselines only ever see the
scenario's *expected* activity curve; the oracle realizes engagement by applying a
mean-preserving multiplicative noise term per (channel, slot) and the same fatigue model.

The objective is transparent and written out in code (`schedule_value`): additive over
placements, each term being `activity * fatigue_multiplier(gap)`.

## Consequences

* Benchmarks are fully reproducible and auditable; `manifest.json` records a checksum of
  the generated scenarios.
* No API keys, no network, no hidden state. Tests run anywhere.
* The headline numbers measure scheduling *skill under a stated model*, not real-world
  engagement. This is called out in the README's "Known gaps": a different, validated
  activity/fatigue model could change the ranking, and real platforms are noisier and
  non-stationary in ways this oracle does not capture.
* Because the oracle adds noise the planner cannot see, no foresight-free strategy is
  perfect, which keeps the comparison honest.
