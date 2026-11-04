import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from scripts.three_issue_validation import assess_phase
from scripts.three_issue_validation import _replay_module, BUNDLE, MODELS
import scripts.three_issue_validation as controller
import scripts.three_issue_throughput_diagnostics as throughput_diagnostics

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


def _throughput_record(exit_code=0):
    return {
        "exit_code": exit_code,
        "process_returncode": exit_code,
        "timed_out": False,
    }


def _throughput_stdout(summary="10 passed", first="110.0", repeated="120.0"):
    return (
        f"Throughput: {first} tok/s\n"
        f"Throughput: {repeated} tok/s\n"
        f"================ {summary} in 8.0s ================\n"
    )


def test_throughput_diagnostic_counterbalances_fresh_process_source_order():
    forward, reverse = throughput_diagnostics.diagnostic_orders()
    assert forward == ("baseline", "common", "850", "859", "793")
    assert reverse == tuple(reversed(forward))


def test_throughput_diagnostic_repeats_the_original_file_without_filters():
    command = throughput_diagnostics.diagnostic_command("python")
    assert command[:3] == ["python", "-m", "pytest"]
    assert command.count("tests/test_continuous_batching.py") == 2
    assert "--keep-duplicates" in command
    assert "--durations=0" in command
    assert not {"-k", "-m", "--deselect", "--ignore"}.intersection(command[3:])


def test_throughput_diagnostic_never_claims_acceptance_from_a_pass():
    result = throughput_diagnostics.summarize_process(
        _throughput_record(), _throughput_stdout()
    )
    assert result["diagnostic_complete"]
    assert result["underlying_tests_passed"]
    assert result["accepted"] is False
    assert result["observations"] == [
        {"invocation": "first", "printed_tokens_per_second": 110.0},
        {"invocation": "repeated", "printed_tokens_per_second": 120.0},
    ]


def test_throughput_diagnostic_retains_failures_when_repeated_invocation_is_fast():
    result = throughput_diagnostics.summarize_process(
        _throughput_record(1), _throughput_stdout("1 failed, 9 passed", first="90.0")
    )
    assert result["diagnostic_complete"]
    assert result["underlying_tests_passed"] is False
    assert result["accepted"] is False
    assert "1 failed" in result["pytest_summary"]


def test_throughput_diagnostic_rejects_metrics_contradicting_a_passing_assertion():
    with pytest.raises(ValueError, match="inconsistent"):
        throughput_diagnostics.summarize_process(
            _throughput_record(), _throughput_stdout(first="90.0")
        )


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0"])
def test_throughput_diagnostic_rejects_invalid_observations(value):
    with pytest.raises(ValueError, match="observation"):
        throughput_diagnostics.summarize_process(
            _throughput_record(), _throughput_stdout(first=value)
        )


@pytest.mark.parametrize(
    "stdout",
    [
        "Throughput: 110.0 tok/s\n========== 10 passed in 8s ==========\n",
        "Throughput: 110.0 tok/s\nThroughput: 120.0 tok/s\n",
        _throughput_stdout("5 passed, 5 skipped"),
        _throughput_stdout("1 failed, 9 passed"),
    ],
)
def test_throughput_diagnostic_rejects_missing_or_inconsistent_completion(stdout):
    with pytest.raises(ValueError):
        throughput_diagnostics.summarize_process(_throughput_record(), stdout)


@pytest.mark.parametrize(
    "changes",
    [
        {"timed_out": True, "exit_code": 124, "process_returncode": -15},
        {"exit_code": -11, "process_returncode": -11},
        {"exit_code": 0, "process_returncode": 1},
    ],
)
def test_throughput_diagnostic_rejects_timeout_crash_or_mismatched_status(changes):
    record = {**_throughput_record(), **changes}
    with pytest.raises(ValueError):
        throughput_diagnostics.summarize_process(record, _throughput_stdout())


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
                        failures = [
                            f"cycle {event['cycle']}: {name}"
                            for event in events
                            for name, ok in event["checks"].items()
                            if not ok
                        ]
                        events.append(
                            {
                                "event": "summary",
                                "outcome": "passed" if passed else "failed",
                                "failures": failures,
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


def _native_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "replay_sources", _replay_module())
    spec = importlib.util.spec_from_file_location(
        "native_cache_probe", BUNDLE / "run_native.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native_cache_inventory(monkeypatch):
    return _native_module(monkeypatch).suite_cache_inventory


def _gpu_measurement_records(tmp_path):
    records = _measurement_records(tmp_path)
    for record in records:
        record["argv"] += ["--memory-accounting", "gpu-footprint"]
        log = Path(record["stdout"])
        events = [json.loads(line) for line in log.read_text().splitlines()]
        failures = []
        diagnostic_failures = []
        for event in events[:-1]:
            event["memory_accounting"] = "gpu-footprint"
            event["diagnostics"] = {"rss_drop": False, "rss_plateau": False}
            for name in ("rss_drop", "rss_plateau"):
                del event["checks"][name]
            failures += [
                f"cycle {event['cycle']}: {name}"
                for name, ok in event["checks"].items()
                if not ok
            ]
            diagnostic_failures += [
                f"cycle {event['cycle']}: rss_drop",
                f"cycle {event['cycle']}: rss_plateau",
            ]
        events[-1].update(
            memory_accounting="gpu-footprint",
            accounting_policy_version=2,
            failures=failures,
            strict_rss_outcome="failed",
            strict_rss_failures=failures + diagnostic_failures,
        )
        log.write_text("\n".join(json.dumps(event) for event in events))
    return records


def test_gpu_accounting_accepts_release_and_reports_failed_strict_diagnostics(tmp_path):
    result = assess_phase(
        "measurements",
        _gpu_measurement_records(tmp_path),
        tmp_path,
        memory_accounting="gpu-footprint",
    )
    assert result["accepted"] is True
    assert result["memory_accounting"] == "gpu-footprint"
    assert result["strict_rss_candidate_failures"] == 24


def test_gpu_accounting_cannot_silently_replace_strict_policy(tmp_path):
    with pytest.raises(ValueError, match="accounting"):
        assess_phase("measurements", _gpu_measurement_records(tmp_path), tmp_path)


@pytest.mark.parametrize("missing", ["footprint_drop", "pool_release", "rss_drop"])
def test_gpu_accounting_rejects_missing_required_or_diagnostic_evidence(
    tmp_path, missing
):
    records = _gpu_measurement_records(tmp_path)
    log = Path(records[1]["stdout"])
    events = [json.loads(line) for line in log.read_text().splitlines()]
    del events[0]["diagnostics" if missing == "rss_drop" else "checks"][missing]
    log.write_text("\n".join(json.dumps(event) for event in events))
    with pytest.raises(ValueError, match="predicates|diagnostics"):
        assess_phase(
            "measurements", records, tmp_path, memory_accounting="gpu-footprint"
        )


def test_gpu_accounting_rejects_falsely_passing_strict_diagnostics(tmp_path):
    records = _gpu_measurement_records(tmp_path)
    log = Path(records[1]["stdout"])
    events = [json.loads(line) for line in log.read_text().splitlines()]
    events[-1]["strict_rss_outcome"] = "passed"
    log.write_text("\n".join(json.dumps(event) for event in events))
    with pytest.raises(ValueError, match="strict"):
        assess_phase(
            "measurements", records, tmp_path, memory_accounting="gpu-footprint"
        )


def test_baseline_cannot_hide_failed_predicates_in_its_summary(tmp_path):
    records = _measurement_records(tmp_path)
    log = Path(records[0]["stdout"])
    events = [json.loads(line) for line in log.read_text().splitlines()]
    events[-1]["failures"] = ["cycle 1: active_drop"]
    log.write_text("\n".join(json.dumps(event) for event in events))
    with pytest.raises(ValueError, match="failures"):
        assess_phase("measurements", records, tmp_path)


@pytest.mark.parametrize("phase", ["suites", "focused", "measurements"])
def test_native_command_selects_full_suite_budget_and_explicit_accounting(
    tmp_path, phase
):
    argv = controller._native_command(tmp_path, phase)
    assert argv[:2] == [sys.executable, str(BUNDLE / "run_native.py")]
    assert argv[argv.index("--phase") + 1] == phase
    assert argv[argv.index("--operator") + 1] == str(tmp_path / "operator.json")
    if phase == "suites":
        assert argv[argv.index("--suite-timeout-seconds") + 1] == "600"
    else:
        assert "--suite-timeout-seconds" not in argv
    if phase == "measurements":
        assert argv[argv.index("--measurement-profile") + 1] == "acceptance48"
        assert argv[argv.index("--memory-accounting") + 1] == "gpu-footprint"
    else:
        assert "--memory-accounting" not in argv


def test_native_runner_records_explicit_bounded_budget(tmp_path, monkeypatch):
    module = _native_module(monkeypatch)
    record = module.run(
        [sys.executable, "-c", "print('complete')"],
        tmp_path,
        tmp_path / "complete",
        dict(os.environ),
        bounded=True,
        timeout_seconds=2,
    )
    assert record["exit_code"] == 0 and record["timed_out"] is False
    assert record["timeout_seconds"] == 2
    assert Path(record["stdout"]).read_text().strip() == "complete"


def test_native_runner_still_terminates_a_real_timeout(tmp_path, monkeypatch):
    module = _native_module(monkeypatch)
    record = module.run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path,
        tmp_path / "timeout",
        dict(os.environ),
        bounded=True,
        timeout_seconds=0.05,
    )
    assert record["exit_code"] == 124 and record["timed_out"] is True
    assert record["timeout_seconds"] == 0.05
    assert record["kill_grace_seconds"] == 10


def _fixture_hub_cache(tmp_path, *, shared):
    cache = tmp_path / "hub"
    declaration = {}
    weights = []
    for model_id, revision in MODELS.items():
        repository = cache / ("models--" + model_id.replace("/", "--"))
        snapshot = repository / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (repository / "refs").mkdir()
        (repository / "refs/main").write_text(revision)
        declaration[model_id] = {"revision": revision}
        for name, data in (
            ("config.json", b"{}"),
            ("tokenizer.json", b'{"version":"1.0"}'),
            ("tokenizer_config.json", b'{"model_max_length":32}'),
            ("model.safetensors", b"small hash-verification fixture"),
        ):
            is_weight = name.endswith(".safetensors")
            etag = (
                hashlib.sha256(data).hexdigest()
                if is_weight
                else hashlib.sha1(
                    b"blob " + str(len(data)).encode() + b"\0" + data
                ).hexdigest()
            )
            blob = repository / "blobs" / etag
            blob.parent.mkdir(exist_ok=True)
            if shared and is_weight:
                xet_hash = hashlib.sha256(b"xet identity " + data).hexdigest()
                target = cache / "blobs" / xet_hash[:2] / xet_hash
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                blob.symlink_to(os.path.relpath(target, blob.parent))
            else:
                blob.write_bytes(data)
            link = snapshot / name
            link.symlink_to(os.path.relpath(blob, snapshot))
            if is_weight:
                weights.append(link)
    return cache, declaration, weights


@pytest.mark.parametrize("shared", [False, True])
def test_native_cache_accepts_authenticated_repo_and_shared_blobs(
    tmp_path, monkeypatch, shared
):
    inventory = _native_cache_inventory(monkeypatch)
    cache, declaration, weights = _fixture_hub_cache(tmp_path, shared=shared)
    result = inventory(cache, declaration)
    for model_id, weight in zip(MODELS, weights):
        assert (
            result[model_id]["files"]["model.safetensors"]
            == hashlib.sha256(weight.read_bytes()).hexdigest()
        )


def test_native_cache_rejects_corrupted_shared_blob(tmp_path, monkeypatch):
    inventory = _native_cache_inventory(monkeypatch)
    cache, declaration, weights = _fixture_hub_cache(tmp_path, shared=True)
    weights[0].resolve().write_bytes(b"corrupted weight fixture")
    with pytest.raises(AssertionError):
        inventory(cache, declaration)


def test_native_cache_rejects_blob_outside_declared_cache(tmp_path, monkeypatch):
    inventory = _native_cache_inventory(monkeypatch)
    cache, declaration, weights = _fixture_hub_cache(tmp_path, shared=False)
    outside = tmp_path / "outside"
    outside.write_bytes(weights[0].read_bytes())
    weights[0].unlink()
    weights[0].symlink_to(outside)
    with pytest.raises(AssertionError):
        inventory(cache, declaration)
