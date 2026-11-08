# slotwise

**An offline, approval-gated send-slot optimizer for social-media posts.**

Given a fixed set of posts per channel and each channel's audience-activity curve,
`slotwise` chooses *when* to publish — which time slots, under per-day and per-channel
caps — to maximize a transparent, simulated-engagement objective that penalizes posting
too often on the same channel (audience fatigue). It then benchmarks its slot choices
against naive, random and greedy baselines on a held-out simulated oracle, and reports
only measured numbers with confidence intervals and a significance test.

Everything runs **offline and deterministically**: no network at runtime, no API keys,
seeded synthetic data. The one place the tool would ever "act" — `publish()` — is a
dry-run that refuses to produce an outbox without an explicit human approval.

---

## How this differs from a social-media *content* agent

This project was written in the same problem area as, but is deliberately distinct from,
LLM-driven social-media **content** agents (which draft posts, apply banned-topic
guardrails, and queue drafts for approval). Those tools answer *what to say*; they
typically have no scheduling model and report no benchmarks.

`slotwise` answers a different, higher-level question — *when to send what you already
have* — and treats it as an optimization problem with a measured answer:

| | content agent (reference area) | **slotwise** |
|---|---|---|
| Core task | draft/guard post text | choose send slots under caps + fatigue |
| Method | LLM + prompt orchestration | exact per-channel dynamic program |
| Objective | implicit (human judgement) | explicit, written-out engagement function |
| Benchmarks | none | efficiency vs 3 baselines, CIs, permutation p |
| Runtime deps | LLM APIs, keys, network | numpy + pydantic, fully offline |
| "Act" step | queue for approval | approval-gated **dry-run**, never networks |

It also does not overlap other projects in this portfolio (DAG scheduling, CI triage,
uplift targeting, retrieval/RAG, working memory, ops triage): the subject is social-media
send-time selection and the engine is a bespoke scheduling DP.

---

## The model (transparent by design)

A **scenario** is a planning horizon of `days × slots_per_day` time slots and a set of
channels. Each channel has an expected per-slot activity curve, a number of posts to
place, and a per-day cap.

The engagement of placing a post on channel `c` at slot `t`, whose previous same-channel
post was `gap` slots earlier, is:

```
value(t, gap) = activity[c][t] * fatigue(gap)
fatigue(gap)  = floor + (1 - floor) * min(gap, recover) / recover      (first post: 1.0)
```

The objective is the sum of `value` over all placements — nothing hidden. Because channels
are independent and fatigue only looks back one post, each channel is solved **exactly**
by a left-to-right dynamic program (see `docs/adr/0002`). The DP's optimality is certified
in tests against a brute-force reference.

### The oracle (why the numbers are honest)

The planner and every baseline see only the *expected* activity. The **oracle** — the
ground truth a schedule is scored against — realizes engagement by multiplying each slot
by seeded, mean-preserving noise (`E[noise] = 1`) and applying the same fatigue model.
So "best in expectation" is not "best in hindsight": even the exact optimizer does not
reach the perfect-foresight maximum, which is why no strategy scores 100%. We report each
strategy's **efficiency** = realized score ÷ perfect-foresight optimum (the same DP run on
realized activity).

---

## Benchmark results

Measured by `slotwise bench` (200 seeded scenarios, 7×24 horizon, noise σ=0.45, 10 000
bootstrap resamples, 20 000 permutations). Reproduce with `slotwise bench`; raw numbers in
[`benchmarks/results.json`](benchmarks/results.json) and run metadata in
[`benchmarks/manifest.json`](benchmarks/manifest.json).

**Efficiency (realized ÷ perfect-foresight optimum), mean over 200 scenarios:**

| strategy | mean efficiency | 95% bootstrap CI | mean realized score |
|---|---|---|---|
| **slotwise** | **0.6379** | [0.6304, 0.6456] | 24.165 |
| greedy | 0.5714 | [0.5639, 0.5790] | 21.621 |
| fixed_time | 0.4233 | [0.4130, 0.4334] | 16.208 |
| random | 0.3335 | [0.3253, 0.3417] | 12.816 |

**slotwise vs each baseline (paired, per-scenario efficiency gain):**

| baseline | mean gain | 95% bootstrap CI | win rate | permutation p |
|---|---|---|---|---|
| greedy | +0.0665 | [0.0569, 0.0762] | 0.830 | 0.00005 |
| fixed_time | +0.2146 | [0.2031, 0.2265] | 1.000 | 0.00005 |
| random | +0.3044 | [0.2937, 0.3151] | 1.000 | 0.00005 |

The meaningful comparison is **slotwise vs greedy**: a real but modest **+6.6 efficiency
points**, winning **83%** of scenarios — not all of them, because the oracle's unseen noise
sometimes rewards greedy's clustering by luck. The `p = 0.00005` value is the floor of a
20 000-permutation test (`(0+1)/(20000+1)`), reported honestly as a floor rather than
"0". The win rates of 1.000 against `fixed_time` and `random` are expected — those are
weak reference points, not close contests.

**Where the gain comes from.** Greedy grabs the highest-activity hours, which on a broad
mid-day plateau sit next to each other, so it clusters posts and eats a fatigue penalty.
`slotwise` trades a little raw activity for spacing. From `slotwise demo --seed 7`:

```
slotwise slots on 'tiktok': [8, 12, 32, 35, 37, 56, 59, 61, 80, 84, 104, 108, 130, 154]
greedy   slots on 'tiktok': [9, 10, 11, 33, 34, 35, 57, 58, 59, 81, 82, 83, 106, 107]
```

---

## Install

Requires Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # runtime deps are just numpy + pydantic
```

## Use

```bash
slotwise bench                 # full benchmark; writes benchmarks/results.json + manifest.json
slotwise bench --no-write      # print the table only
slotwise demo --seed 7         # optimize one scenario, compare strategies, show spacing
slotwise publish --seed 7      # REFUSED: the approval gate blocks an unapproved send
slotwise publish --seed 7 --confirm --approved-by khadija   # dry-run outbox, no network
```

As a library:

```python
from slotwise import optimize_schedule, publish, Approval
from slotwise.generate import generate_scenario

scenario = generate_scenario(seed=7)
schedule = optimize_schedule(scenario)
result = publish(scenario, schedule, approval=Approval(approved_by="khadija", confirmed=True))
print(result.outbox[0])   # "[DRY-RUN] ... publish on day 0 at slot ..."
```

---

## Project layout

```
src/slotwise/
  models.py      # pydantic v2 model: Channel, Scenario, Schedule, Approval, fatigue
  generate.py    # seeded synthetic scenario generator
  optimize.py    # exact per-channel DP + brute-force reference (for tests)
  oracle.py      # simulated-engagement oracle + perfect-foresight upper bound
  baselines.py   # fixed_time, random, greedy
  stats.py       # bootstrap CI, paired permutation test, win rate
  benchmark.py   # harness, results.json + manifest.json
  publish.py     # approval-gated, dry-run publish
  cli.py         # argparse CLI: bench / demo / publish
tests/           # 127 tests (pytest + Hypothesis), incl. the approval-gate safety tests
docs/adr/        # architecture decision records
benchmarks/      # committed results.json + manifest.json from a real run
```

## Known gaps

This is an honest list of what the numbers do **not** prove.

- **The oracle is a model, not reality.** Results measure scheduling skill under a stated
  activity-and-fatigue model with lognormal noise. A different, validated model — real
  platforms are noisier, non-stationary, and audience-dependent — could change the
  ranking. slotwise has never been run against live engagement data.
- **Greedy is fatigue-blind by construction.** It is the standard myopic baseline. A
  fatigue-aware greedy would close much of the +6.6-point gap; we did not claim to beat a
  spacing-aware heuristic, only the common one.
- **Channels are assumed independent.** There is no cross-channel audience overlap or
  shared cap, which is what makes the per-channel DP exact. Add either and the exactness
  claim no longer holds.
- **First-order fatigue only.** The penalty depends on the single preceding post, not a
  rolling window of recent posts.
- **Posts within a channel are interchangeable scheduling units** (no per-post quality or
  per-post release windows). Adding heterogeneous post weights would make the exact
  decomposition a heuristic.
- **`publish` is a dry-run.** There are no real platform adapters; the approval gate is
  designed so that adding one later cannot bypass human oversight, but that integration
  does not exist yet.

## License

MIT © 2026 Khadija. See [LICENSE](LICENSE).
