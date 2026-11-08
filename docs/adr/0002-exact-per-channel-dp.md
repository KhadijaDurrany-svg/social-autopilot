# ADR 0002: Exact per-channel dynamic program

## Status

Accepted.

## Context

The scheduler must choose slots to maximize a fatigue-aware objective under per-day caps.
Posting-time optimization with a fatigue penalty is, in general, a constrained assignment
problem. We wanted the optimizer to be *provably* optimal for the stated objective so the
benchmark reflects "the best achievable under the model" rather than another heuristic
whose gaps we could not characterize. We also wanted to avoid heavy solver dependencies
(no MILP/ortools) to keep the install small and offline.

## Decision

Exploit two properties of the model:

1. **Channels are independent** -- a post's engagement depends only on its own channel's
   activity and the spacing of that channel's posts. So the problem decomposes per
   channel.
2. **Fatigue is first-order** -- the penalty on a placement depends only on the gap to the
   immediately preceding placement on the same channel.

Together these make each channel solvable exactly by a left-to-right dynamic program over
slots, with state `(posts_placed, posts_today, gap_since_last)`. The gap is clamped at the
fatigue recovery horizon because any larger gap behaves identically (multiplier 1.0),
keeping the state space small.

## Consequences

* The optimizer is exact for the stated objective. This is certified in tests against a
  brute-force reference over hundreds of randomized instances (including a Hypothesis
  property test).
* The same DP computes the perfect-foresight upper bound by running it on *realized*
  activity, giving a principled denominator for the efficiency metric.
* The decomposition assumes channels do not compete for the same audience at the same
  time. If cross-channel fatigue or shared caps were added, this exactness would no longer
  hold and the solver would need to change (noted in "Known gaps").
* Higher-order fatigue (depending on several previous posts) would enlarge the state and
  is out of scope.
