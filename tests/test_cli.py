from __future__ import annotations

import json

import pytest

from slotwise.cli import main


def test_cli_bench_no_write(capsys):
    rc = main(["bench", "--scenarios", "6", "--n-boot", "200", "--n-perm", "200", "--no-write"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "slotwise" in out
    assert "p_value" in out


def test_cli_bench_writes(tmp_path, capsys):
    out_dir = tmp_path / "bench"
    rc = main(
        ["bench", "--scenarios", "5", "--n-boot", "200", "--n-perm", "200", "--out", str(out_dir)]
    )
    assert rc == 0
    results = json.loads((out_dir / "results.json").read_text())
    assert results["config"]["n_scenarios"] == 5
    assert (out_dir / "manifest.json").exists()


def test_cli_demo(capsys):
    rc = main(["demo", "--seed", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "perfect-foresight optimum" in out
    assert "efficiency" in out


def test_cli_publish_refused(capsys):
    rc = main(["publish", "--seed", "3"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err


def test_cli_publish_confirmed(capsys):
    rc = main(["publish", "--seed", "3", "--confirm", "--approved-by", "khadija"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "approved_by=khadija" in out
    assert "DRY-RUN" in out


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])
