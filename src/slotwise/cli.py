"""Command-line interface for slotwise.

Subcommands
-----------
* ``bench``   -- run the benchmark; writes ``benchmarks/results.json`` and
  ``manifest.json`` unless ``--no-write`` is given.
* ``demo``    -- generate one scenario, optimize it, and print the schedule next to the
  baselines with their oracle scores.
* ``publish`` -- demonstrate the approval gate; refuses to act unless ``--confirm`` and
  ``--approved-by`` are supplied.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slotwise import __version__
from slotwise.baselines import fixed_time, greedy, random_schedule
from slotwise.benchmark import format_table, run_benchmark, write_outputs
from slotwise.generate import generate_scenario
from slotwise.models import Approval, ApprovalRequired
from slotwise.optimize import optimize_schedule
from slotwise.oracle import Oracle
from slotwise.publish import publish


def _cmd_bench(args: argparse.Namespace) -> int:
    results = run_benchmark(
        n_scenarios=args.scenarios,
        seed=args.seed,
        sigma=args.sigma,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
    )
    print(format_table(results))
    if not args.no_write:
        out_dir = Path(args.out)
        results_path, manifest_path = write_outputs(results, out_dir)
        print(f"\nwrote {results_path}")
        print(f"wrote {manifest_path}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    scenario = generate_scenario(args.seed)
    oracle = Oracle(scenario, seed=args.seed, sigma=args.sigma)
    best = oracle.best()
    schedules = {
        "slotwise": optimize_schedule(scenario),
        "greedy": greedy(scenario),
        "fixed_time": fixed_time(scenario),
        "random": random_schedule(scenario, seed=args.seed),
    }
    print(f"scenario seed={args.seed} channels={[c.name for c in scenario.channels]}")
    print(f"perfect-foresight optimum = {best:.3f}\n")
    print(f"{'strategy':<12}{'oracle_score':>14}{'efficiency':>12}")
    for name, sched in schedules.items():
        sc = oracle.score(sched)
        eff = sc / best if best > 0 else 0.0
        print(f"{name:<12}{sc:>14.3f}{eff:>12.4f}")
    first = scenario.channels[0].name
    print(f"\nslotwise slots on {first!r}: {schedules['slotwise'].slots_for(first)}")
    print(f"greedy   slots on {first!r}: {schedules['greedy'].slots_for(first)}")
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    scenario = generate_scenario(args.seed)
    schedule = optimize_schedule(scenario)
    approval = None
    if args.confirm:
        approval = Approval(approved_by=args.approved_by or "", confirmed=True)
    try:
        result = publish(scenario, schedule, approval=approval)
    except ApprovalRequired as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"approved_by={result.approved_by} dry_run={result.dry_run} placements={len(result.placements)}")
    for line in result.outbox[: args.limit]:
        print(line)
    if len(result.outbox) > args.limit:
        print(f"... ({len(result.outbox) - args.limit} more)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slotwise", description=__doc__)
    parser.add_argument("--version", action="version", version=f"slotwise {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser("bench", help="run the benchmark")
    p_bench.add_argument("--scenarios", type=int, default=200)
    p_bench.add_argument("--seed", type=int, default=20260101)
    p_bench.add_argument("--sigma", type=float, default=0.45)
    p_bench.add_argument("--n-boot", type=int, default=10000, dest="n_boot")
    p_bench.add_argument("--n-perm", type=int, default=20000, dest="n_perm")
    p_bench.add_argument("--no-write", action="store_true", help="do not write results/manifest")
    p_bench.add_argument("--out", default="benchmarks", help="output directory")
    p_bench.set_defaults(func=_cmd_bench)

    p_demo = sub.add_parser("demo", help="optimize one scenario and compare strategies")
    p_demo.add_argument("--seed", type=int, default=7)
    p_demo.add_argument("--sigma", type=float, default=0.45)
    p_demo.set_defaults(func=_cmd_demo)

    p_pub = sub.add_parser("publish", help="demonstrate the approval-gated (dry-run) publish")
    p_pub.add_argument("--seed", type=int, default=7)
    p_pub.add_argument("--confirm", action="store_true", help="supply explicit confirmation")
    p_pub.add_argument("--approved-by", default="", dest="approved_by")
    p_pub.add_argument("--limit", type=int, default=10)
    p_pub.set_defaults(func=_cmd_publish)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
