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


def diagnostic_command(python):
    return [
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


def run_diagnostics(root):
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
        "purpose": "Diagnose the unchanged greater-than-100 throughput assertion without altering acceptance semantics",
        "source_manifest_sha256": runner.digest(controller.BUNDLE / "manifest.json"),
        "diagnostic_script_sha256": runner.digest(Path(__file__).resolve()),
        "workflow_commit": os.environ.get("GITHUB_SHA"),
        "operator_sha256": runner.digest(root / "operator.json"),
        "orders": diagnostic_orders(),
        "records": [],
    }
    try:
        for order_index, order in enumerate(diagnostic_orders(), 1):
            for variant in order:
                controller._verify_sources(root)
                record = runner.run(
                    diagnostic_command(python),
                    root / variant,
                    output / f"order-{order_index}-{variant}",
                    dict(environment, PYTHONPATH=str(root / variant)),
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
                except ValueError as error:
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
    raise SystemExit(run_diagnostics(parser.parse_args().run_root.resolve()))
