"""Run frozen issue candidates through the existing native acceptance protocol."""

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / ".github/validation/three-issue"
MODELS = {
    "mlx-community/Llama-3.2-1B-Instruct-4bit": "08231374eeacb049a0eade7922910865b8fce912",
    "mlx-community/Qwen3-0.6B-8bit": "11de96878523501bcaa86104e3c186de07ff9068",
}
# Match mlx-lm's model/tokenizer download inputs; repository prose is not a
# runtime prerequisite and may be absent from an otherwise complete cache.
MODEL_FILES = [
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
]
CHECKS = {
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
}
FOCUSED_FILES = (
    (
        "850",
        (
            "test_scheduler_sampling_controls.py",
            "test_mllm_media_specprefill.py",
            "test_request.py",
            "test_batch_compat.py",
        ),
    ),
    (
        "859",
        (
            "test_gpt_oss_tool_parsing.py",
            "test_harmony_parsers.py",
            "test_engine_base.py",
        ),
    ),
    (
        "793",
        (
            "test_model_registry.py",
            "test_engine_core_request_owner.py",
            "test_server_cache_controls.py",
            "test_memory_cache_mlx.py",
        ),
    ),
    ("793", ("test_server_cache_controls.py", "test_engine_core_request_owner.py")),
    ("793", ("test_mllm_scheduler_stop_release.py",)),
    ("793", ("test_memory_cache_mlx.py",)),
)


def _argument(argv: list[str], flag: str) -> str:
    if argv.count(flag) != 1:
        raise ValueError(f"Expected exactly one {flag}")
    return argv[argv.index(flag) + 1]


def _memory_accounting(record: dict) -> str:
    argv = record["argv"]
    policy = (
        _argument(argv, "--memory-accounting")
        if "--memory-accounting" in argv
        else "strict-rss"
    )
    if policy not in {"strict-rss", "gpu-footprint"}:
        raise ValueError("Unknown memory accounting policy")
    return policy


def _measurement_key(record: dict, memory_accounting: str) -> tuple[str, str, str, str]:
    argv = record["argv"]
    if _memory_accounting(record) != memory_accounting:
        raise ValueError("Measurement accounting policy mismatch")
    for flag, value in (
        ("--cycles", "3"),
        ("--kv-mib", "256"),
        ("--cache-limit-mib", "1024"),
        ("--settle-seconds", "60"),
        ("--requests", "3"),
        ("--prompt-repeats", "1024"),
    ):
        if _argument(argv, flag) != value:
            raise ValueError(f"Measurement protocol changed at {flag}")
    if "--synthetic-registry" in argv:
        raise ValueError("Unexpected measurement matrix mode")
    mode = (
        "production-registry"
        if "--models-config" in argv
        else "production" if "--model" in argv else "synthetic"
    )
    return (
        record["variant"],
        mode,
        _argument(argv, "--pre-delete-barrier"),
        _argument(argv, "--route"),
    )


def _check_measurement_log(record: dict, run_root: Path) -> str:
    log = Path(record["stdout"]).resolve()
    if not log.is_relative_to(run_root.resolve()):
        raise ValueError("Measurement log escapes run output")
    events = []
    for line in log.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    summaries = [event for event in events if event.get("event") == "summary"]
    cycles = [event for event in events if event.get("event") == "cycle_checks"]
    if len(summaries) != 1 or [event.get("cycle") for event in cycles] != [1, 2, 3]:
        raise ValueError("Missing terminal measurement evidence")
    policy = _memory_accounting(record)
    rss_checks = {"rss_drop", "rss_plateau"}
    required = CHECKS - rss_checks if policy == "gpu-footprint" else CHECKS
    if any(
        set(event.get("checks", {})) != required
        or any(type(value) is not bool for value in event["checks"].values())
        or event.get("memory_accounting", "strict-rss") != policy
        for event in cycles
    ):
        raise ValueError("Incomplete terminal measurement predicates")
    if any(
        set(event.get("diagnostics", {}))
        != (rss_checks if policy == "gpu-footprint" else set())
        or any(
            type(value) is not bool for value in event.get("diagnostics", {}).values()
        )
        for event in cycles
    ):
        raise ValueError("Incomplete RSS diagnostics")
    failures = [
        f"cycle {event['cycle']}: {name}"
        for event in cycles
        for name, ok in event["checks"].items()
        if not ok
    ]
    diagnostic_failures = [
        f"cycle {event['cycle']}: {name}"
        for event in cycles
        for name, ok in event.get("diagnostics", {}).items()
        if not ok
    ]
    passed = all(all(event["checks"].values()) for event in cycles)
    summary = summaries[0]
    if summary.get("memory_accounting", "strict-rss") != policy:
        raise ValueError("Summary accounting policy mismatch")
    if policy == "gpu-footprint" and summary.get("accounting_policy_version") != 2:
        raise ValueError("Missing accounting policy version")
    expected_outcome = "passed" if passed else "failed"
    if summary.get("outcome") != expected_outcome or record["exit_code"] != (
        0 if passed else 1
    ):
        raise ValueError("Inconsistent terminal measurement result")
    if summary.get("failures") != failures:
        raise ValueError("Measurement summary hides or misreports failures")
    strict_outcome = "failed" if failures or diagnostic_failures else "passed"
    if policy == "gpu-footprint" or "strict_rss_outcome" in summary:
        if (
            summary.get("strict_rss_outcome") != strict_outcome
            or summary.get("strict_rss_failures") != failures + diagnostic_failures
        ):
            raise ValueError("Inconsistent strict RSS diagnostic result")
    return strict_outcome


def assess_phase(
    phase: str,
    records: list[dict],
    run_root: Path,
    *,
    memory_accounting: str = "strict-rss",
) -> dict:
    """Check phase completion; final acceptance also requires reviewing skips."""
    if phase == "suites":
        if Counter(row["variant"] for row in records) != Counter(
            ("baseline", "common", "850", "859", "793")
        ):
            raise ValueError("Incomplete suite matrix")
    elif phase == "focused":
        observed = Counter(
            (
                row["variant"],
                tuple(
                    Path(arg).name for arg in row["argv"] if arg.startswith("tests/")
                ),
            )
            for row in records
        )
        if observed != Counter(FOCUSED_FILES):
            raise ValueError("Incomplete focused matrix")
    elif phase == "measurements":
        expected = Counter(
            {
                (variant, mode, barrier, route): 3
                for variant in ("baseline", "793")
                for mode, barrier in (
                    ("synthetic", "owner-streams"),
                    ("production", "owner-streams"),
                    ("production-registry", "owner-streams"),
                    ("production-registry", "none"),
                )
                for route in ("prefix", "all")
            }
        )
        if (
            Counter(_measurement_key(row, memory_accounting) for row in records)
            != expected
        ):
            raise ValueError("Incomplete measurement matrix")
    else:
        raise ValueError(f"Unknown phase {phase}")
    candidates = [row for row in records if row["variant"] != "baseline"]
    if any(row["exit_code"] != 0 or row["timed_out"] for row in candidates):
        raise ValueError("A candidate failed or timed out")
    baselines = [row for row in records if row["variant"] == "baseline"]
    if any(
        row["timed_out"] or row["exit_code"] not in {0, 1, 2, 3} for row in baselines
    ):
        raise ValueError("A baseline crashed or did not produce a normal pytest result")
    strict_candidate_failures = 0
    if phase == "measurements":
        for row in records:
            if row["timed_out"]:
                raise ValueError("Missing terminal measurement evidence after timeout")
            strict = _check_measurement_log(row, run_root)
            strict_candidate_failures += (
                row["variant"] != "baseline" and strict == "failed"
            )
    return {
        "accepted": True,
        "memory_accounting": memory_accounting if phase == "measurements" else None,
        "strict_rss_candidate_failures": strict_candidate_failures,
        "candidate_runs": len(candidates),
        "baseline_exit_codes": [
            row["exit_code"] for row in records if row["variant"] == "baseline"
        ],
        "scope": "Phase checks only; inspect test skips and provenance before final acceptance",
    }


def _replay_module():
    spec = importlib.util.spec_from_file_location(
        "three_issue_replay", BUNDLE / "replay_sources.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_sources(root: Path) -> None:
    replay = _replay_module()
    replay.verify_bundle()
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    for variant in ("baseline", "common", "850", "859", "793"):
        path = (
            manifest["base_files"]
            if variant == "baseline"
            else (
                manifest["common"]
                if variant == "common"
                else manifest["issues"][variant]
            )["expected_files"]
        )
        expected = json.loads((BUNDLE / path).read_text())
        if variant == "baseline":
            expected["scripts/benchmark_cache_release.py"] = {
                "sha256": manifest["benchmark_sha256"],
                "untracked": True,
                "mode": manifest["benchmark_mode"],
            }
        replay.verify_tree(root / variant, expected, allow_runtime_artifacts=True)


def provision(root: Path, existing_cache: Path | None) -> None:
    from huggingface_hub import snapshot_download

    cache = existing_cache.resolve() if existing_cache else root / "hub"
    inputs = root / "inputs"
    inputs.mkdir()
    residents = {}
    for name, (model_id, revision) in zip(("first", "second"), MODELS.items()):
        model = Path(
            snapshot_download(
                model_id,
                revision=revision,
                cache_dir=str(cache),
                allow_patterns=MODEL_FILES,
                local_files_only=existing_cache is not None,
            )
        ).resolve()
        ref = cache / ("models--" + model_id.replace("/", "--")) / "refs/main"
        if existing_cache:
            if ref.read_text().strip() != revision:
                raise ValueError("Existing fixture revision does not match")
        else:
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(revision)
        weights = inputs / f"{name}-weights.json"
        subprocess.run(
            [
                sys.executable,
                str(BUNDLE / "inventory_model.py"),
                "--repo",
                str(REPO),
                "--model",
                str(model),
                "--output",
                str(weights),
            ],
            check=True,
        )
        residents[name] = {
            "model_path": str(model),
            "revision": revision,
            "weights_manifest": str(weights),
        }
    # JSON is valid YAML and avoids adding a second serializer dependency here.
    models_config = inputs / "models.yaml"
    models_config.write_text(
        json.dumps(
            {
                "manager": {
                    "memory_budget_gb": 3,
                    "idle_unload_seconds": 0,
                    "contention_policy": {"strategy": "fail"},
                },
                "models": [
                    {
                        "name": name,
                        "path": row["model_path"],
                        "continuous_batching": True,
                        "estimated_memory_gb": 1,
                    }
                    for name, row in residents.items()
                ],
            },
            indent=2,
        )
        + "\n"
    )
    first = residents["first"]
    operator = {
        "repo": str(REPO),
        "run_root": str(root),
        "python": sys.executable,
        "hf_hub_cache": str(cache),
        "suite_models": {
            model_id: {"revision": revision} for model_id, revision in MODELS.items()
        },
        "benchmark_model": first["model_path"],
        "benchmark_model_revision": first["revision"],
        "benchmark_weights_manifest": first["weights_manifest"],
        "models_config": str(models_config),
        "resident_models": list(residents),
        "registry_artifacts": residents,
    }
    (root / "operator.json").write_text(json.dumps(operator, indent=2) + "\n")


def _native_command(root: Path, phase: str) -> list[str]:
    command = [
        sys.executable,
        str(BUNDLE / "run_native.py"),
        "--operator",
        str(root / "operator.json"),
        "--phase",
        phase,
    ]
    if phase == "measurements":
        command += [
            "--measurement-profile",
            "acceptance48",
            "--memory-accounting",
            "gpu-footprint",
        ]
    elif phase == "suites":
        command += ["--suite-timeout-seconds", "600"]
    elif phase != "focused":
        raise ValueError(f"Unknown phase {phase}")
    return command


def run_phase(root: Path, phase: str) -> None:
    _verify_sources(root)
    receipts = root / "receipts"
    receipts.mkdir(exist_ok=True)
    before = set((root / "results").glob(f"*-{phase}"))
    command = _native_command(root, phase)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTEST_ADDOPTS", None)
    with (
        (receipts / f"{phase}.stdout").open("wb") as stdout,
        (receipts / f"{phase}.stderr").open("wb") as stderr,
    ):
        process = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr)
    _verify_sources(root)
    new = set((root / "results").glob(f"*-{phase}")) - before
    if len(new) != 1 or not ((directory := new.pop()) / "results.json").is_file():
        raise ValueError(
            f"No terminal {phase} results; inspect receipts, runner exit {process.returncode}"
        )
    payload = json.loads((directory / "results.json").read_text())
    if payload["phase"] != phase:
        raise ValueError("Phase provenance mismatch")
    assessment = assess_phase(
        phase,
        payload["records"],
        root,
        memory_accounting="gpu-footprint" if phase == "measurements" else "strict-rss",
    )
    assessment.update(
        runner_exit_code=process.returncode,
        results=str(directory),
        source_manifests_verified_before_and_after=True,
    )
    (receipts / f"{phase}-assessment.json").write_text(
        json.dumps(assessment, indent=2) + "\n"
    )
    print(json.dumps(assessment, indent=2))


def static_checks(root: Path) -> None:
    _verify_sources(root)
    for variant in ("common", "850", "859", "793"):
        for command in (
            [
                "ruff",
                "check",
                "vllm_mlx/",
                "tests/",
                "scripts/client_acceptance/",
                "--select",
                "E,F,W",
                "--ignore",
                "E402,E501,E731,F811,F841",
            ],
            ["black", "--check", "vllm_mlx/", "tests/", "scripts/client_acceptance/"],
            ["mypy", "vllm_mlx/", "--ignore-missing-imports", "--no-error-summary"],
        ):
            subprocess.run(
                [sys.executable, "-m", *command], cwd=root / variant, check=True
            )
    _verify_sources(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "provision", "run", "static"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("suites", "focused", "measurements"))
    parser.add_argument("--existing-hub-cache", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not root.is_relative_to(REPO) or root == REPO:
        parser.error("Use a dedicated run root inside this checkout")
    if args.operation == "prepare":
        _replay_module().materialize(REPO, root)
        print(f"Verified frozen sources at {root}")
    elif args.operation == "provision":
        provision(root, args.existing_hub_cache)
    elif args.operation == "static":
        static_checks(root)
    elif not args.phase:
        parser.error("run requires --phase")
    else:
        run_phase(root, args.phase)


if __name__ == "__main__":
    main()
