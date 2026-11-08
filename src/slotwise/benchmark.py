"""Benchmark harness.

For each seeded scenario we build every strategy's schedule, score it on the scenario's
oracle, and record its efficiency (realized score / perfect-foresight optimum). We then
aggregate across scenarios with bootstrap confidence intervals, and compare the optimizer
against each baseline with a paired permutation test and a win rate.

The numbers written to ``benchmarks/results.json`` are *measured*, never hand-written.
``manifest.json`` captures the git sha (or ``"n/a"``), Python version, and a checksum of
the scenario inputs so a run can be reproduced and audited.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from slotwise import baselines
from slotwise.generate import generate_scenario
from slotwise.optimize import optimize_schedule
from slotwise.oracle import Oracle
from slotwise.stats import bootstrap_mean_ci, paired_permutation_pvalue, win_rate

STRATEGIES = ("slotwise", "greedy", "fixed_time", "random")


@dataclass(frozen=True)
class StrategyStat:
    strategy: str
    mean_efficiency: float
    ci_lo: float
    ci_hi: float
    mean_score: float


@dataclass(frozen=True)
class Comparison:
    baseline: str
    mean_efficiency_gain: float
    gain_ci_lo: float
    gain_ci_hi: float
    win_rate: float
    p_value: float


def _build_schedules(scenario, scenario_seed: int) -> dict[str, object]:
    return {
        "slotwise": optimize_schedule(scenario),
        "greedy": baselines.greedy(scenario),
        "fixed_time": baselines.fixed_time(scenario),
        "random": baselines.random_schedule(scenario, seed=scenario_seed),
    }


def run_benchmark(
    n_scenarios: int = 200,
    seed: int = 20260101,
    days: int = 7,
    slots_per_day: int = 24,
    sigma: float = 0.45,
    n_boot: int = 10000,
    n_perm: int = 20000,
) -> dict:
    """Run the benchmark and return a JSON-serializable results dict."""
    rng = np.random.default_rng(seed)
    scenario_seeds = [int(x) for x in rng.integers(0, 2**31 - 1, size=n_scenarios)]

    # efficiency[strategy] = per-scenario realized/optimal; scores[strategy] = raw realized.
    efficiency: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    scores: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    checksum = hashlib.sha256()

    for ssd in scenario_seeds:
        scenario = generate_scenario(ssd, days=days, slots_per_day=slots_per_day)
        checksum.update(scenario.model_dump_json().encode("utf-8"))
        oracle = Oracle(scenario, seed=ssd, sigma=sigma)
        best = oracle.best()
        schedules = _build_schedules(scenario, ssd)
        for strat in STRATEGIES:
            sched = schedules[strat]
            sched.validate_against(scenario)  # defensive: every schedule must be feasible
            sc = oracle.score(sched)
            scores[strat].append(sc)
            efficiency[strat].append(sc / best if best > 0 else 0.0)

    eff_arr = {s: np.asarray(v, dtype=float) for s, v in efficiency.items()}
    score_arr = {s: np.asarray(v, dtype=float) for s, v in scores.items()}

    strat_stats: list[StrategyStat] = []
    for s in STRATEGIES:
        mean, lo, hi = bootstrap_mean_ci(eff_arr[s], n_boot=n_boot, seed=seed + 1)
        strat_stats.append(
            StrategyStat(
                strategy=s,
                mean_efficiency=mean,
                ci_lo=lo,
                ci_hi=hi,
                mean_score=float(score_arr[s].mean()),
            )
        )

    comparisons: list[Comparison] = []
    opt = eff_arr["slotwise"]
    for base in ("greedy", "fixed_time", "random"):
        diffs = opt - eff_arr[base]
        mean, lo, hi = bootstrap_mean_ci(diffs, n_boot=n_boot, seed=seed + 2)
        p = paired_permutation_pvalue(diffs, n_perm=n_perm, seed=seed + 3)
        comparisons.append(
            Comparison(
                baseline=base,
                mean_efficiency_gain=mean,
                gain_ci_lo=lo,
                gain_ci_hi=hi,
                win_rate=win_rate(diffs),
                p_value=p,
            )
        )

    return {
        "config": {
            "n_scenarios": n_scenarios,
            "seed": seed,
            "days": days,
            "slots_per_day": slots_per_day,
            "sigma": sigma,
            "n_boot": n_boot,
            "n_perm": n_perm,
        },
        "strategies": [asdict(s) for s in strat_stats],
        "comparisons": [asdict(c) for c in comparisons],
        "dataset_checksum": checksum.hexdigest(),
    }


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else "n/a"
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def build_manifest(results: dict) -> dict:
    return {
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "dataset_checksum": results["dataset_checksum"],
        "config": results["config"],
    }


def write_outputs(results: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    manifest_path = out_dir / "manifest.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    manifest_path.write_text(json.dumps(build_manifest(results), indent=2, sort_keys=True) + "\n")
    return results_path, manifest_path


def format_table(results: dict) -> str:
    """Render a compact human-readable summary of ``results``."""
    lines: list[str] = []
    cfg = results["config"]
    lines.append(
        f"scenarios={cfg['n_scenarios']} seed={cfg['seed']} sigma={cfg['sigma']} "
        f"horizon={cfg['days']}x{cfg['slots_per_day']}"
    )
    lines.append("")
    lines.append(f"{'strategy':<12}{'mean_eff':>10}{'95% CI':>22}{'mean_score':>14}")
    for s in results["strategies"]:
        ci = f"[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]"
        lines.append(
            f"{s['strategy']:<12}{s['mean_efficiency']:>10.4f}{ci:>22}{s['mean_score']:>14.3f}"
        )
    lines.append("")
    lines.append("slotwise vs baseline (efficiency gain):")
    lines.append(f"{'baseline':<12}{'gain':>10}{'95% CI':>22}{'win_rate':>10}{'p_value':>12}")
    for c in results["comparisons"]:
        ci = f"[{c['gain_ci_lo']:.4f}, {c['gain_ci_hi']:.4f}]"
        lines.append(
            f"{c['baseline']:<12}{c['mean_efficiency_gain']:>10.4f}{ci:>22}"
            f"{c['win_rate']:>10.3f}{c['p_value']:>12.5f}"
        )
    return "\n".join(lines)


__all__ = [
    "run_benchmark",
    "build_manifest",
    "write_outputs",
    "format_table",
    "STRATEGIES",
]


if __name__ == "__main__":  # pragma: no cover - convenience entrypoint
    res = run_benchmark()
    print(format_table(res))
    sys.exit(0)
