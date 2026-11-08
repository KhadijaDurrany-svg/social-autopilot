from __future__ import annotations

import json

from slotwise.benchmark import (
    STRATEGIES,
    build_manifest,
    format_table,
    run_benchmark,
    write_outputs,
)


def test_run_benchmark_structure():
    r = run_benchmark(n_scenarios=12, n_boot=500, n_perm=500, seed=1)
    assert {s["strategy"] for s in r["strategies"]} == set(STRATEGIES)
    assert {c["baseline"] for c in r["comparisons"]} == {"greedy", "fixed_time", "random"}
    assert len(r["dataset_checksum"]) == 64


def test_benchmark_optimizer_leads():
    r = run_benchmark(n_scenarios=40, n_boot=800, n_perm=800, seed=2)
    eff = {s["strategy"]: s["mean_efficiency"] for s in r["strategies"]}
    assert eff["slotwise"] >= eff["greedy"]
    assert eff["greedy"] >= eff["random"]
    gains = {c["baseline"]: c["mean_efficiency_gain"] for c in r["comparisons"]}
    assert gains["greedy"] > 0.0
    assert gains["random"] > 0.0


def test_benchmark_pvalues_in_range():
    r = run_benchmark(n_scenarios=20, n_boot=500, n_perm=1000, seed=3)
    for c in r["comparisons"]:
        assert 0.0 < c["p_value"] <= 1.0
        assert 0.0 <= c["win_rate"] <= 1.0


def test_write_outputs_and_manifest(tmp_path):
    r = run_benchmark(n_scenarios=8, n_boot=300, n_perm=300, seed=4)
    rp, mp = write_outputs(r, tmp_path / "out")
    assert rp.exists() and mp.exists()
    manifest = json.loads(mp.read_text())
    assert set(manifest) >= {"git_sha", "python_version", "numpy_version", "dataset_checksum", "config"}
    assert manifest["dataset_checksum"] == r["dataset_checksum"]
    # results.json round-trips.
    loaded = json.loads(rp.read_text())
    assert loaded["config"]["n_scenarios"] == 8


def test_build_manifest_git_sha_is_string():
    r = run_benchmark(n_scenarios=4, n_boot=200, n_perm=200, seed=5)
    m = build_manifest(r)
    assert isinstance(m["git_sha"], str) and m["git_sha"]


def test_format_table_mentions_strategies():
    r = run_benchmark(n_scenarios=6, n_boot=200, n_perm=200, seed=6)
    txt = format_table(r)
    for strat in STRATEGIES:
        assert strat in txt


def test_benchmark_reproducible():
    a = run_benchmark(n_scenarios=10, n_boot=300, n_perm=300, seed=99)
    b = run_benchmark(n_scenarios=10, n_boot=300, n_perm=300, seed=99)
    assert a["dataset_checksum"] == b["dataset_checksum"]
    assert a["strategies"] == b["strategies"]
    assert a["comparisons"] == b["comparisons"]
