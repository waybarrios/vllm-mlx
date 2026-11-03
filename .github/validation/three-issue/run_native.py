#!/usr/bin/env python3
"""Run the unchanged native gates with explicit operator inputs and raw provenance."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

from replay_sources import BUNDLE, digest, verify_bundle, verify_tree


SUITE_MODELS = {
    "mlx-community/Llama-3.2-1B-Instruct-4bit": ["tests/test_batching_deterministic.py", "tests/test_engine_core_stream_safety.py"],
    "mlx-community/Qwen3-0.6B-8bit": ["tests/test_continuous_batching.py", "tests/test_streaming_detokenizer.py"],
}
FULL_SUITE = ["-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider"]


def measurement_matrix(profile):
    historical = (("synthetic", "owner-streams"), ("production", "owner-streams"))
    profiles = {
        "historical24": historical,
        "acceptance48": historical + (("production-registry", "owner-streams"), ("production-registry", "none")),
        "exhaustive96": tuple((mode, barrier) for mode in ("synthetic", "production", "synthetic-registry", "production-registry") for barrier in ("owner-streams", "none")),
    }
    return tuple((mode, barrier, route, repeat, variant) for mode, barrier in profiles[profile] for route in ("prefix", "all") for repeat in (1, 2, 3) for variant in ("baseline", "793"))


def native_environment(cache, output):
    # The existing Hub cache is read input; offline mode prevents model fetching.
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_CACHE=str(cache), TRANSFORMERS_CACHE=str(cache))
    for variable in ("HF_HOME", "HF_ASSETS_CACHE", "HF_MODULES_CACHE", "HF_DATASETS_CACHE", "TMPDIR", "XDG_CACHE_HOME", "BLACK_CACHE_DIR", "RUFF_CACHE_DIR", "MYPY_CACHE_DIR"):
        location = output / variable.lower()
        location.mkdir()
        environment[variable] = str(location)
    return environment


def verify_local_model(root, identity):
    root = Path(root).expanduser().resolve()
    declared = Path(identity["weights_manifest"]).expanduser().resolve()
    files = json.loads(declared.read_text())["files"]
    actual_paths = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    assert actual_paths == set(files) and "config.json" in files
    assert any(name.endswith(".safetensors") for name in files)
    for relative, expected in files.items():
        path = root / relative
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        assert digest(path) == expected, relative
    assert identity["revision"]
    return {"model_path": str(root), "revision": identity["revision"], "weights_manifest_sha256": digest(declared), "file_count": len(files)}


def suite_cache_inventory(cache, declarations):
    assert set(declarations) == set(SUITE_MODELS), "Declare the exact configured fixture IDs, not benchmark replacements"
    result = {}
    for model_id in SUITE_MODELS:
        revision = declarations[model_id]["revision"]
        assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
        repository = cache / ("models--" + model_id.replace("/", "--"))
        assert (repository / "refs/main").read_text().strip() == revision
        snapshot = repository / "snapshots" / revision
        assert snapshot.is_dir() and (snapshot / "config.json").is_file()
        files = {}
        for path in sorted(snapshot.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            assert resolved.is_relative_to(cache), str(path)
            value = digest(path)
            # Standard Hub cache blob names authenticate the cached bytes locally.
            blob = resolved.name
            if len(blob) == 64:
                assert value == blob, str(path)
            elif len(blob) == 40:
                data = path.read_bytes()
                assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == blob, str(path)
            else:
                raise AssertionError(f"Expected an existing content-addressed Hub cache blob: {path}")
            files[str(path.relative_to(snapshot))] = value
        assert any(name.endswith(".safetensors") for name in files)
        assert "tokenizer_config.json" in files and "tokenizer.json" in files
        result[model_id] = {"revision": revision, "snapshot": str(snapshot), "files": files, "fixture_sources": SUITE_MODELS[model_id]}
    return result


def run(argv, cwd, stem, environment, *, bounded):
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    with stem.with_suffix(".stdout").open("wb") as stdout, stem.with_suffix(".stderr").open("wb") as stderr:
        process = subprocess.Popen(argv, cwd=cwd, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        killed = False
        try:
            code = process.wait(timeout=120 if bounded else None)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                killed = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 137 if killed else 124
    record = {"argv": argv, "cwd": str(cwd), "started_at_utc": started_at, "exit_code": code, "process_returncode": process.returncode, "duration_seconds": round(time.monotonic() - started, 3), "timeout_seconds": 120 if bounded else None, "kill_grace_seconds": 10 if bounded else None, "timed_out": timed_out, "stdout": str(stem.with_suffix(".stdout")), "stderr": str(stem.with_suffix(".stderr"))}
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--phase", choices=("suites", "focused", "measurements"), required=True)
    parser.add_argument("--measurement-profile", choices=("historical24", "acceptance48", "exhaustive96"), default="acceptance48", help="acceptance48 is the default; exhaustive96 is optional diagnostic coverage")
    args = parser.parse_args()
    verify_bundle()
    assert platform.system() == "Darwin" and platform.machine() == "arm64", "Actual Apple Silicon macOS required"
    config = json.loads(args.operator.read_text())
    repo, root, python = (Path(config[key]).expanduser().resolve() for key in ("repo", "run_root", "python"))
    assert root.is_relative_to(repo) and root.is_dir() and python.is_file() and os.access(python, os.X_OK)
    assert "VLLM_MLX_TEST_MODEL" not in os.environ, "Do not override fixture model identities"
    assert not os.environ.get("PYTEST_ADDOPTS"), "Use the frozen pytest defaults without injected filters"
    cache = Path(config["hf_hub_cache"]).expanduser().resolve()
    assert cache.is_dir(), "Use an existing provisioned cache as read input; no fetching or copying"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "results" / f"{stamp}-{args.phase}"
    output.mkdir(parents=True)
    environment = native_environment(cache, output)
    device_code = "import json,platform,mlx.core as mx; assert platform.system()=='Darwin' and platform.machine()=='arm64'; assert mx.metal.is_available(); print(json.dumps(getattr(mx,'device_info',mx.metal.device_info)()))"
    subprocess.run([str(python), "-c", device_code], env=environment, check=True, stdout=(output / "metal-device.json").open("w"))
    versions_code = "import importlib.metadata,json,sys; print(json.dumps({'python':sys.version,'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()))},indent=2))"
    subprocess.run([str(python), "-c", versions_code], env=environment, check=True, stdout=(output / "installed-versions.json").open("w"))
    subprocess.run(["sw_vers"], env=environment, check=True, stdout=(output / "os-build.txt").open("w"))
    subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"], env=environment, check=True, stdout=(output / "chip-and-ram.txt").open("w"))
    cache_identity = suite_cache_inventory(cache, config["suite_models"])
    (output / "suite-model-cache.json").write_text(json.dumps(cache_identity, indent=2) + "\n")
    bundle = json.loads((BUNDLE / "manifest.json").read_text())
    for variant in ("common", "850", "859", "793"):
        layer = bundle["common"] if variant == "common" else bundle["issues"][variant]
        verify_tree(root / variant, json.loads((BUNDLE / layer["expected_files"]).read_text()), allow_runtime_artifacts=True)
    base_expected = json.loads((BUNDLE / bundle["base_files"]).read_text())
    baseline_files = {name: row for name, row in base_expected.items()}
    baseline_files["scripts/benchmark_cache_release.py"] = {"sha256": bundle["benchmark_sha256"], "untracked": True, "mode": bundle["benchmark_mode"]}
    verify_tree(root / "baseline", baseline_files, allow_runtime_artifacts=True)
    model = Path(config["benchmark_model"]).expanduser().resolve()
    models_config = Path(config["models_config"]).expanduser().resolve()
    assert (model / "config.json").is_file() and models_config.is_file()
    assert config["benchmark_model_revision"] and len(set(config["resident_models"])) >= 2
    benchmark_identity = verify_local_model(model, {"revision": config["benchmark_model_revision"], "weights_manifest": config["benchmark_weights_manifest"]})
    assert set(config["registry_artifacts"]) == set(config["resident_models"])
    registry_identities = {name: verify_local_model(row["model_path"], row) for name, row in config["registry_artifacts"].items()}
    registry_code = "import json,pathlib,sys,yaml; p=pathlib.Path(sys.argv[1]); c=yaml.safe_load(p.read_text()); print(json.dumps({m['name']:str(pathlib.Path(m['path']).expanduser().resolve()) for m in c['models']}))"
    paths = json.loads(subprocess.check_output([str(python), "-c", registry_code, str(models_config)], env=environment))
    assert all(paths[name] == row["model_path"] for name, row in registry_identities.items())
    provenance = {"operator_input_sha256": digest(args.operator), "base": bundle["base"], "bundle_manifest_sha256": digest(BUNDLE / "manifest.json"), "python": str(python), "platform": platform.platform(), "benchmark_model": benchmark_identity, "registry_config_sha256": digest(models_config), "registry_models": registry_identities, "resident_models": config["resident_models"], "optional_fixture_sources": bundle["optional_fixture_sources"], "fixture_model_substitutions": False, "measurement_profile": args.measurement_profile if args.phase == "measurements" else None, "full_suite_and_metal_acceptance": "PENDING until terminal logs and all unchanged predicates are independently reviewed"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    records = []

    def execute(variant, label, argv, bounded=True):
        child_environment = dict(environment, PYTHONPATH=str(root / variant))
        record = run([str(python), *argv], root / variant, output / label, child_environment, bounded=bounded)
        records.append({"variant": variant, **record})

    if args.phase == "suites":
        for variant in ("baseline", "common", "850", "859", "793"):
            execute(variant, f"{variant}-full-configured-suite", FULL_SUITE)
    elif args.phase == "focused":
        execute("850", "850-focused", ["-m", "pytest", "tests/test_scheduler_sampling_controls.py", "tests/test_mllm_media_specprefill.py", "tests/test_request.py", "tests/test_batch_compat.py", "-q"])
        execute("859", "859-focused", ["-m", "pytest", "tests/test_gpt_oss_tool_parsing.py", "tests/test_harmony_parsers.py", "tests/test_engine_base.py", "-q"])
        execute("793", "793-registry", ["-m", "pytest", "tests/test_model_registry.py", "tests/test_engine_core_request_owner.py", "tests/test_server_cache_controls.py", "tests/test_memory_cache_mlx.py", "-q"])
        # Preserve the three commands in the original native protocol as well.
        for label, files in (("793-contracts", ["tests/test_server_cache_controls.py", "tests/test_engine_core_request_owner.py"]), ("793-shutdown", ["tests/test_mllm_scheduler_stop_release.py"]), ("793-native-memory", ["tests/test_memory_cache_mlx.py"])):
            execute("793", label, ["-m", "pytest", *files, "-q", "--tb=short"])
    else:
        for mode, barrier, route, repeat, variant in measurement_matrix(args.measurement_profile):
            mode_args = []
            if mode == "production":
                mode_args = ["--model", str(model)]
            elif mode == "synthetic-registry":
                mode_args = ["--synthetic-registry"]
            elif mode == "production-registry":
                mode_args = ["--models-config", str(models_config)]
                for name in config["resident_models"]:
                    mode_args += ["--resident-model", name]
            argv = ["scripts/benchmark_cache_release.py", *mode_args, "--pre-delete-barrier", barrier, "--route", route, "--cycles", "3", "--kv-mib", "256", "--cache-limit-mib", "1024", "--settle-seconds", "60", "--requests", "3", "--prompt-repeats", "1024"]
            execute(variant, f"{variant}-{mode}-{barrier}-{route}-{repeat}", argv, bounded=False)
    (output / "results.json").write_text(json.dumps({"phase": args.phase, "measurement_profile": args.measurement_profile if args.phase == "measurements" else None, "provenance": provenance, "records": records, "acceptance": "NOT automatically established; inspect all terminal summaries, skips, failures and unchanged measurement gates"}, indent=2) + "\n")
    print(json.dumps({"results": str(output), "runs": len(records), "nonzero_runs": sum(row["exit_code"] != 0 for row in records)}, indent=2))
    return int(any(row["exit_code"] != 0 for row in records))


if __name__ == "__main__":
    raise SystemExit(main())
