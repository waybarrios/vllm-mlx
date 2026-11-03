import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from scripts.three_issue_validation import assess_phase
from scripts.three_issue_validation import _replay_module, BUNDLE, MODELS

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


def _native_cache_inventory(monkeypatch):
    monkeypatch.setitem(sys.modules, "replay_sources", _replay_module())
    spec = importlib.util.spec_from_file_location(
        "native_cache_probe", BUNDLE / "run_native.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.suite_cache_inventory


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
