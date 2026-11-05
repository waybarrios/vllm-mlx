"""Portable contracts for the paired native performance report."""

import copy
import os
import sys

import pytest

from scripts.performance_ci.controller import (
    _model_manifest,
    pair_order,
    run_child,
    summarize,
)

CASE = {
    "id": "tiny-decode",
    "model_id": "example/tiny",
    "revision": "a" * 40,
    "prompt_tokens": 8,
    "output_tokens": 4,
    "concurrency": 1,
    "prefix_cache": False,
}
SOURCES = {"base": "b" * 40, "head": "c" * 40}


def test_model_manifest_hashes_snapshots_inside_default_hub_cache(tmp_path):
    snapshot = tmp_path / ".cache" / "huggingface" / "hub" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"abc")
    metadata = snapshot / ".cache" / "huggingface"
    metadata.mkdir(parents=True)
    (metadata / "download.metadata").write_text("not model content")

    assert _model_manifest(snapshot) == {
        "config.json": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    }


def observations():
    records = []
    for repeat, (base, head) in enumerate([(100, 110), (200, 180)]):
        for side, speed in [("base", base), ("head", head)]:
            metrics = {
                "batch_seconds": 4 / speed,
                "ttft_seconds": 0.01,
                "tokens_per_second": speed,
                "peak_mlx_bytes": 1024,
                "output_tokens": 4,
                "cached_tokens": 0,
                "output_sha256": "d" * 64,
            }
            records.append(
                {
                    "side": side,
                    "repeat": repeat,
                    "case": copy.deepcopy(CASE),
                    "source_sha": SOURCES[side],
                    "prompt_sha256": "e" * 64,
                    "runtime": {"mlx": "0.32.4", "python": "3.12.10"},
                    "model_load_seconds": 1.0,
                    "phases": {"first": metrics, "warm": metrics.copy()},
                }
            )
    return records


def test_order_alternates_to_balance_time_drift():
    assert pair_order(0) == ("base", "head")
    assert pair_order(1) == ("head", "base")
    assert pair_order(2) == ("base", "head")


def test_report_uses_paired_changes_and_retains_variability():
    report = summarize(observations(), [CASE], 2, SOURCES)
    row = next(
        r
        for r in report["rows"]
        if r["phase"] == "warm" and r["metric"] == "tokens_per_second"
    )
    assert row["base_median"] == 150
    assert row["head_median"] == 145
    assert row["base_mad"] == 50
    assert row["paired_change_percent"] == pytest.approx(0)
    assert row["pairs"] == 2
    assert report["performance_gate"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected"])
def test_report_rejects_incomplete_or_extra_trials(mutation):
    records = observations()
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(copy.deepcopy(records[0]))
    else:
        records[0]["repeat"] = 9
    with pytest.raises(ValueError):
        summarize(records, [CASE], 2, SOURCES)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_report_rejects_invalid_measurements(value):
    records = observations()
    records[0]["phases"]["warm"]["ttft_seconds"] = value
    with pytest.raises(ValueError):
        summarize(records, [CASE], 2, SOURCES)


@pytest.mark.parametrize("field", ["prompt_sha256", "runtime", "source_sha", "case"])
def test_report_rejects_incompatible_workload_or_provenance(field):
    records = observations()
    records[1][field] = "incompatible"
    with pytest.raises(ValueError):
        summarize(records, [CASE], 2, SOURCES)


def test_report_rejects_short_generation():
    records = observations()
    records[0]["phases"]["warm"]["output_tokens"] = 3
    with pytest.raises(ValueError):
        summarize(records, [CASE], 2, SOURCES)


def test_report_exposes_output_changes_without_gating_performance():
    records = observations()
    records[1]["phases"]["warm"]["output_sha256"] = "f" * 64
    report = summarize(records, [CASE], 2, SOURCES)
    assert any("output" in warning.lower() for warning in report["warnings"])


def test_child_preserves_failure_output(tmp_path):
    log = tmp_path / "failed.log"
    with pytest.raises(RuntimeError):
        run_child(
            [sys.executable, "-c", "print('failure evidence'); raise SystemExit(7)"],
            log,
            5,
            os.environ.copy(),
        )
    assert "failure evidence" in log.read_text()


def test_child_timeout_kills_process_group_and_retains_log(tmp_path):
    log = tmp_path / "hung.log"
    pid_file = tmp_path / "pid"
    command = [
        sys.executable,
        "-c",
        "import subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"open({str(pid_file)!r}, 'w').write(str(p.pid)); "
        "print('started', flush=True); time.sleep(60)",
    ]
    with pytest.raises(TimeoutError):
        run_child(command, log, 0.5, os.environ.copy())
    assert "started" in log.read_text()
    # Linux can retain a killed grandchild briefly as a zombie until reaped.
    proc_stat = __import__("pathlib").Path(f"/proc/{pid_file.read_text()}/stat")
    if proc_stat.exists():
        assert proc_stat.read_text().split()[2] == "Z"
