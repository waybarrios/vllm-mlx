"""Collect separate repeated-invocation diagnostics without changing acceptance."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


def diagnostic_orders():
    variants = ("baseline", "common", "850", "859", "793")
    return variants, tuple(reversed(variants))


def diagnostic_command(python, *, profile=False):
    command = [
        str(python),
        "-m",
        "pytest",
        "tests/test_continuous_batching.py",
        "tests/test_continuous_batching.py",
        "--keep-duplicates",
        "-q",
        "-s",
        "--tb=short",
        "--durations=0",
        "-p",
        "no:cacheprovider",
    ]
    if profile:
        command += ["-p", "three_issue_throughput_profile"]
    return command


def summarize_profile(profiles, test_sha256, source_hashes):
    required = {
        "scheduler.step",
        "scheduler.admit",
        "scheduler.outputs",
        "scheduler.cleanup",
        "generator.next",
        "generator.prefill",
        "generator.start_decode",
        "generator.decode",
        "generator.forward",
        "engine.add_request",
        "engine.cleanup",
        "collector.put",
        "collector.get",
    }
    if len(profiles) != 2:
        raise ValueError("Missing complete first/repeated phase profiles")
    for profile in profiles:
        calls = profile.get("calls", [])
        ids = {call.get("id") for call in calls}
        if (
            profile.get("accepted") is not False
            or profile.get("outcome") not in {"passed", "failed"}
            or profile.get("test_sha256") != test_sha256
            or not required.issubset(call.get("name") for call in calls)
            or len(ids) != len(calls)
            or any(
                type(call.get("id")) is not int
                or type(call.get("thread_id")) is not int
                or call.get("parent_id") == call["id"]
                or (call.get("parent_id") is not None and call["parent_id"] not in ids)
                or any(
                    type(call.get(field)) not in (int, float)
                    or not math.isfinite(call[field])
                    or call[field] < 0
                    for field in ("start_seconds", "wall_seconds", "thread_cpu_seconds")
                )
                for call in calls
            )
            or any(
                profile.get("sources", {}).get(name, {}).get("sha256") != digest
                for name, digest in source_hashes.items()
            )
        ):
            raise ValueError("Missing, mismatched or incomplete phase profile")
    return {"profiles_complete": True, "accepted": False, "invocations": len(profiles)}


def summarize_process(record, stdout):
    if (
        record["timed_out"]
        or record["exit_code"] not in (0, 1)
        or record["exit_code"] != record["process_returncode"]
    ):
        raise ValueError("Diagnostic process timed out, crashed or has no normal exit")
    values = re.findall(r"^Throughput: (\S+) tok/s$", stdout, re.M)
    try:
        values = [float(value) for value in values]
    except ValueError as error:
        raise ValueError("Invalid diagnostic observation") from error
    if len(values) != 2 or not all(
        math.isfinite(value) and value > 0 for value in values
    ):
        raise ValueError("Missing or invalid diagnostic observation")
    summaries = re.findall(r"^=+ (.*?\b\d+ passed.*?) =+$", stdout, re.M)
    if len(summaries) != 1:
        raise ValueError("Missing terminal pytest summary")
    counts = dict(
        (name, int(count))
        for count, name in re.findall(
            r"\b(\d+) (passed|failed|skipped|error|errors|deselected)\b", summaries[0]
        )
    )
    if (
        set(counts) - {"passed", "failed"}
        or sum(counts.values()) != 10
        or bool(counts.get("failed", 0)) != (record["exit_code"] == 1)
        or (record["exit_code"] == 0 and any(value < 100 for value in values))
    ):
        raise ValueError("Incomplete or inconsistent diagnostic pytest result")
    return {
        "diagnostic_complete": True,
        "underlying_tests_passed": record["exit_code"] == 0,
        "accepted": False,
        "pytest_summary": summaries[0],
        "observations": [
            {"invocation": name, "printed_tokens_per_second": value}
            for name, value in zip(("first", "repeated"), values)
        ],
        "scope": "Separate fresh-process/repeated-invocation diagnostic, not full-suite acceptance or proof of a compiler cause",
    }


def run_diagnostics(root, *, profile=False):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from scripts import three_issue_validation as controller

    assert root.is_relative_to(repo) and root.is_dir()
    assert platform.system() == "Darwin" and platform.machine() == "arm64"
    assert "VLLM_MLX_TEST_MODEL" not in os.environ
    assert not os.environ.get("PYTEST_ADDOPTS")
    controller._verify_sources(root)
    # Reuse the frozen runner's source, cache and bounded-child controls.
    sys.path.insert(0, str(controller.BUNDLE))
    spec = importlib.util.spec_from_file_location(
        "diagnostic_native_runner", controller.BUNDLE / "run_native.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.verify_bundle()
    config = json.loads((root / "operator.json").read_text())
    assert Path(config["repo"]).resolve() == repo
    assert Path(config["run_root"]).resolve() == root
    python = Path(config["python"]).resolve()
    assert python == Path(sys.executable).resolve()
    cache = Path(config["hf_hub_cache"]).resolve()
    output = (
        root / "diagnostics" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    environment = runner.native_environment(cache, output)
    model_cache = runner.suite_cache_inventory(cache, config["suite_models"])
    (output / "suite-model-cache.json").write_text(
        json.dumps(model_cache, indent=2) + "\n"
    )
    device_code = "import json,mlx.core as mx; assert mx.metal.is_available(); print(json.dumps(getattr(mx,'device_info',mx.metal.device_info)()))"
    versions_code = "import importlib.metadata,json,sys; print(json.dumps({'python':sys.version,'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()))},indent=2))"
    for name, command in (
        ("metal-device.json", [str(python), "-c", device_code]),
        ("installed-versions.json", [str(python), "-c", versions_code]),
        ("os-build.txt", ["sw_vers"]),
        (
            "chip-and-ram.txt",
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
        ),
    ):
        with (output / name).open("wb") as stream:
            subprocess.run(command, env=environment, stdout=stream, check=True)
    result = {
        "accepted": False,
        "complete_matrix": False,
        "instrumented": profile,
        "purpose": "Diagnose the unchanged greater-than-100 throughput assertion without altering acceptance semantics",
        "source_manifest_sha256": runner.digest(controller.BUNDLE / "manifest.json"),
        "diagnostic_script_sha256": runner.digest(Path(__file__).resolve()),
        "profile_plugin_sha256": (
            runner.digest(repo / "scripts/three_issue_throughput_profile.py")
            if profile
            else None
        ),
        "workflow_commit": os.environ.get("GITHUB_SHA"),
        "operator_sha256": runner.digest(root / "operator.json"),
        "orders": diagnostic_orders(),
        "records": [],
    }
    try:
        for order_index, order in enumerate(diagnostic_orders(), 1):
            for variant in order:
                controller._verify_sources(root)
                child_environment = dict(environment, PYTHONPATH=str(root / variant))
                profile_path = output / f"order-{order_index}-{variant}.profile.json"
                if profile:
                    child_environment.update(
                        PYTHONPATH=os.pathsep.join(
                            (str(root / variant), str(repo / "scripts"))
                        ),
                        VLLM_MLX_THROUGHPUT_PROFILE=str(profile_path),
                    )
                record = runner.run(
                    diagnostic_command(python, profile=profile),
                    root / variant,
                    output / f"order-{order_index}-{variant}",
                    child_environment,
                    bounded=True,
                    timeout_seconds=120,
                )
                record.update(
                    variant=variant,
                    order=order_index,
                    original_test_sha256=runner.digest(
                        root / variant / "tests/test_continuous_batching.py"
                    ),
                )
                try:
                    record["diagnostic"] = summarize_process(
                        record, Path(record["stdout"]).read_text()
                    )
                    if profile:
                        record["profile"] = summarize_profile(
                            json.loads(profile_path.read_text()),
                            record["original_test_sha256"],
                            {
                                f"vllm_mlx.{name}": runner.digest(
                                    root / variant / "vllm_mlx" / f"{name}.py"
                                )
                                for name in (
                                    "scheduler",
                                    "engine_core",
                                    "output_collector",
                                )
                            },
                        )
                        record["profile_sha256"] = runner.digest(profile_path)
                except (ValueError, OSError) as error:
                    record["diagnostic"] = {
                        "diagnostic_complete": False,
                        "accepted": False,
                        "error": str(error),
                    }
                result["records"].append(record)
                controller._verify_sources(root)
                (output / "results.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
        result["complete_matrix"] = all(
            row["diagnostic"]["diagnostic_complete"] for row in result["records"]
        )
        result["source_manifests_verified_before_and_after"] = True
    finally:
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "results": str(output),
                "diagnostic_only": True,
                "accepted": False,
                "complete_matrix": result["complete_matrix"],
            }
        )
    )
    return int(
        not result["complete_matrix"]
        or any(row["exit_code"] != 0 for row in result["records"])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Record separate instrumented phase timings; never acceptance",
    )
    args = parser.parse_args()
    raise SystemExit(run_diagnostics(args.run_root.resolve(), profile=args.profile))
