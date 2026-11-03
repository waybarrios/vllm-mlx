import json
from pathlib import Path

import pytest

from scripts.three_issue_validation import assess_phase

_CHECK_NAMES = (
    "sufficient_workload",
    "cache_was_populated",
    "cache_entries_cleared",
    "active_drop",
    "pool_release",
    "unused_pool_empty",
    "rss_drop",
    "footprint_drop",
    "rss_plateau",
    "footprint_plateau",
    "idle_steps_unchanged",
    "model_resident",
    "registry_residents_unchanged",
)


def _suite_records():
    return [
        {"variant": variant, "exit_code": 0, "timed_out": False}
        for variant in ("baseline", "common", "850", "859", "793")
    ]


def test_complete_candidate_suites_pass_with_recorded_baseline_failure(tmp_path):
    records = _suite_records()
    records[0]["exit_code"] = 3
    result = assess_phase("suites", records, tmp_path)
    assert result["accepted"] is True
    assert result["baseline_exit_codes"] == [3]
    assert result["candidate_runs"] == 4


@pytest.mark.parametrize(
    ("exit_code", "timed_out"),
    [(124, True), (-11, False), (137, False), (4, False), (5, False), (3, True)],
)
def test_incomplete_baseline_suite_is_rejected(tmp_path, exit_code, timed_out):
    records = _suite_records()
    records[0].update(exit_code=exit_code, timed_out=timed_out)
    with pytest.raises(ValueError, match="baseline"):
        assess_phase("suites", records, tmp_path)


@pytest.mark.parametrize("missing", ["common", "850", "859", "793"])
def test_missing_candidate_suite_is_rejected(tmp_path, missing):
    records = [row for row in _suite_records() if row["variant"] != missing]
    with pytest.raises(ValueError, match="suite matrix"):
        assess_phase("suites", records, tmp_path)


@pytest.mark.parametrize("variant", ["common", "850", "859", "793"])
def test_candidate_suite_timeout_is_rejected(tmp_path, variant):
    records = _suite_records()
    row = next(row for row in records if row["variant"] == variant)
    row.update(exit_code=124, timed_out=True)
    with pytest.raises(ValueError, match="candidate"):
        assess_phase("suites", records, tmp_path)


def _measurement_records(tmp_path, *, baseline_summary=True):
    records = []
    for mode, barrier in (
        ("synthetic", "owner-streams"),
        ("production", "owner-streams"),
        ("production-registry", "owner-streams"),
        ("production-registry", "none"),
    ):
        for route in ("prefix", "all"):
            for repeat in (1, 2, 3):
                for variant in ("baseline", "793"):
                    log = tmp_path / f"{variant}-{mode}-{barrier}-{route}-{repeat}.log"
                    events = []
                    if variant == "793" or baseline_summary:
                        passed = variant == "793"
                        events = [
                            {
                                "event": "cycle_checks",
                                "cycle": cycle,
                                "checks": {key: passed for key in _CHECK_NAMES},
                            }
                            for cycle in (1, 2, 3)
                        ]
                        events.append(
                            {
                                "event": "summary",
                                "outcome": "passed" if passed else "failed",
                                "failures": [] if passed else ["cycle 1: active_drop"],
                            }
                        )
                    log.write_text("\n".join(json.dumps(event) for event in events))
                    argv = [
                        "python",
                        "scripts/benchmark_cache_release.py",
                        "--pre-delete-barrier",
                        barrier,
                        "--route",
                        route,
                        "--cycles",
                        "3",
                        "--kv-mib",
                        "256",
                        "--cache-limit-mib",
                        "1024",
                        "--settle-seconds",
                        "60",
                        "--requests",
                        "3",
                        "--prompt-repeats",
                        "1024",
                    ]
                    if mode == "production":
                        argv += ["--model", "/existing/model"]
                    elif mode == "production-registry":
                        argv += [
                            "--models-config",
                            "/existing/models.yaml",
                            "--resident-model",
                            "first",
                            "--resident-model",
                            "second",
                        ]
                    records.append(
                        {
                            "variant": variant,
                            "exit_code": 0 if variant == "793" else 1,
                            "timed_out": False,
                            "stdout": str(log),
                            "argv": argv,
                        }
                    )
    return records


def test_terminal_baseline_measurement_failures_are_recorded(tmp_path):
    result = assess_phase("measurements", _measurement_records(tmp_path), tmp_path)
    assert result["accepted"] is True
    assert result["candidate_runs"] == 24
    assert result["baseline_exit_codes"] == [1] * 24


def test_baseline_measurement_crash_is_not_an_expected_regression(tmp_path):
    with pytest.raises(ValueError, match="terminal"):
        assess_phase(
            "measurements",
            _measurement_records(tmp_path, baseline_summary=False),
            tmp_path,
        )


def test_candidate_measurement_failure_is_rejected(tmp_path):
    records = _measurement_records(tmp_path)
    records[1]["exit_code"] = 1
    with pytest.raises(ValueError, match="candidate"):
        assess_phase("measurements", records, tmp_path)


def test_missing_measurement_run_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="measurement matrix"):
        assess_phase("measurements", _measurement_records(tmp_path)[:-1], tmp_path)


def test_duplicate_measurement_cannot_replace_missing_mode(tmp_path):
    records = _measurement_records(tmp_path)
    records[-1] = records[1].copy()
    with pytest.raises(ValueError, match="measurement matrix"):
        assess_phase("measurements", records, tmp_path)


def test_missing_focused_group_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="focused matrix"):
        assess_phase("focused", [], tmp_path)


def test_unknown_phase_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown phase"):
        assess_phase("anything", [], Path(tmp_path))
