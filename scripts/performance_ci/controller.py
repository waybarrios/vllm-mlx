"""Portable orchestration and reporting for native performance workers."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys

METRICS = ("tokens_per_second", "ttft_seconds", "batch_seconds", "peak_mlx_bytes")
PHASES = ("first", "warm")


def pair_order(repeat):
    return ("base", "head") if repeat % 2 == 0 else ("head", "base")


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def summarize(records, cases, repeats, sources):
    """Reject incomplete comparisons; performance differences are informational."""
    expected = {
        (case["id"], repeat, side)
        for case in cases
        for repeat in range(repeats)
        for side in ("base", "head")
    }
    indexed = {}
    by_id = {case["id"]: case for case in cases}
    if not expected or len(by_id) != len(cases):
        raise ValueError("Empty or duplicate cases")
    for record in records:
        case = record.get("case")
        if not isinstance(case, dict) or case != by_id.get(case.get("id")):
            raise ValueError("Unexpected workload")
        key = (case["id"], record["repeat"], record["side"])
        if key not in expected or key in indexed:
            raise ValueError(f"Unexpected or duplicate observation {key}")
        if record["source_sha"] != sources[record["side"]]:
            raise ValueError("Wrong source SHA")
        if not _digest(record["prompt_sha256"]):
            raise ValueError("Invalid prompt digest")
        if not isinstance(record["runtime"], dict) or not record["runtime"]:
            raise ValueError("Missing runtime provenance")
        if not _positive(record["model_load_seconds"]):
            raise ValueError("Invalid load time")
        if set(record["phases"]) != set(PHASES):
            raise ValueError("Missing phase")
        for metrics in record["phases"].values():
            if any(not _positive(metrics[m]) for m in METRICS):
                raise ValueError("Invalid measurement")
            if metrics["output_tokens"] != case["output_tokens"] * case["concurrency"]:
                raise ValueError("Wrong output length")
            if not _digest(metrics["output_sha256"]):
                raise ValueError("Invalid output digest")
            if (
                type(metrics["cached_tokens"]) is not int
                or metrics["cached_tokens"] < 0
            ):
                raise ValueError("Invalid cache-hit count")
        indexed[key] = record
    if set(indexed) != expected:
        raise ValueError("Incomplete paired observations")

    rows, warnings = [], set()
    runtime = records[0]["runtime"]
    for case in cases:
        samples = [
            indexed[(case["id"], repeat, side)]
            for repeat in range(repeats)
            for side in ("base", "head")
        ]
        if any(
            s["runtime"] != runtime or s["prompt_sha256"] != samples[0]["prompt_sha256"]
            for s in samples
        ):
            raise ValueError("Incompatible runtime or prompts")
        for phase in PHASES:
            outputs = {s["phases"][phase]["output_sha256"] for s in samples}
            if len(outputs) != 1:
                warnings.add(
                    f"{case['id']} {phase}: output tokens differ; inspect raw results"
                )
            if case["prefix_cache"] and phase == "warm":
                if any(s["phases"][phase]["cached_tokens"] == 0 for s in samples):
                    warnings.add(
                        f"{case['id']}: warmed prefix reuse had zero cache hits"
                    )
        for phase, metric in [("load", "model_load_seconds")] + [
            (p, m) for p in PHASES for m in METRICS
        ]:
            values = {}
            for side in ("base", "head"):
                values[side] = []
                for repeat in range(repeats):
                    record = indexed[(case["id"], repeat, side)]
                    values[side].append(
                        record[metric]
                        if phase == "load"
                        else record["phases"][phase][metric]
                    )
            row = {
                "case": case["id"],
                "phase": phase,
                "metric": metric,
                "pairs": repeats,
            }
            for side in ("base", "head"):
                median = statistics.median(values[side])
                row[f"{side}_median"] = median
                row[f"{side}_mad"] = statistics.median(
                    abs(v - median) for v in values[side]
                )
            row["paired_change_percent"] = statistics.median(
                (head / base - 1) * 100
                for base, head in zip(values["base"], values["head"])
            )
            rows.append(row)
    return {"performance_gate": False, "rows": rows, "warnings": sorted(warnings)}


def run_child(command, log_path, timeout, env):
    """Bound the whole process tree, including grandchildren, retaining output."""
    with Path(log_path).open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException as error:
            # The worker has its own session, so terminal interrupts do not
            # reach it. Retire its group before propagating an interrupted wait.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            if isinstance(error, subprocess.TimeoutExpired):
                raise TimeoutError(
                    f"Worker exceeded {timeout}s; see {log_path}"
                ) from None
            raise
        if code:
            raise RuntimeError(f"Worker exited {code}; see {log_path}")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _source_sha(path):
    sha = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Expected an immutable source commit")
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if dirty:
        raise ValueError(f"Source has tracked modifications: {path}")
    return sha


def _model_manifest(directory):
    files = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and ".cache" not in path.relative_to(directory).parts:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            files[str(path.relative_to(directory))] = digest.hexdigest()
    return files


def markdown(report, sources):
    lines = [
        "## Apple Silicon paired performance",
        "",
        f"Base `{sources['base']}`; candidate `{sources['head']}`.",
        "",
        "Report only. Changes are candidate/base percentages, median over pairs.",
        "Higher tokens/s is faster; lower time/memory is better. ± is median absolute deviation.",
        "First batch uses a fresh process/model, not cold OS/disk caches. Warm repeats the same prompts.",
        "",
        "| Case | Phase | Metric | Base median ± MAD | Candidate median ± MAD | Paired change | Pairs |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['case']} | {row['phase']} | {row['metric']} | "
            f"{row['base_median']:.4g} ± {row['base_mad']:.3g} | "
            f"{row['head_median']:.4g} ± {row['head_mad']:.3g} | "
            f"{row['paired_change_percent']:+.2f}% | {row['pairs']} |"
        )
    lines += ["", *[f"- {w}" for w in report["warnings"]], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--head", required=True, type=Path)
    parser.add_argument("--profile", choices=("quick", "nightly"), default="quick")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary_path = args.output / "summary.md"
    summary_path.write_text(
        "## Apple Silicon paired performance\n\nMeasurements incomplete. Inspect the logs.\n"
    )
    records = []
    try:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Native measurements require macOS arm64")
        from huggingface_hub import snapshot_download

        config = json.loads(Path(__file__).with_name("workloads.json").read_text())
        profile = config["profiles"][args.profile]
        roots = {"base": args.base.resolve(), "head": args.head.resolve()}
        sources = {side: _source_sha(root) for side, root in roots.items()}
        models, cases = {}, []
        for name in profile["models"]:
            model = config["models"][name]
            directory = snapshot_download(
                repo_id=model["id"], revision=model["revision"]
            )
            models[name] = directory
            _write_json(
                args.output / f"model-{name}.json",
                {**model, "files": _model_manifest(directory)},
            )
            for case in config["cases"]:
                cases.append(
                    {
                        **case,
                        "id": f"{name}-{case['id']}",
                        "model_id": model["id"],
                        "revision": model["revision"],
                    }
                )
        _write_json(
            args.output / "provenance.json",
            {
                "sources": sources,
                "profile": args.profile,
                "config": config,
                "platform": platform.platform(),
                "python": sys.version,
                "runner_image": os.environ.get("ImageVersion"),
                "hardware": subprocess.check_output(
                    ["system_profiler", "SPHardwareDataType", "-json"], text=True
                ),
            },
        )
        env = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "0",
        }
        for case in cases:
            name = next(
                n for n in models if case["model_id"] == config["models"][n]["id"]
            )
            for repeat in range(profile["repeats"]):
                for side in pair_order(repeat):
                    stem = f"{case['id']}-{repeat}-{side}"
                    spec_path = args.output / f"{stem}-spec.json"
                    result_path = args.output / f"{stem}.json"
                    _write_json(
                        spec_path,
                        {
                            "side": side,
                            "repeat": repeat,
                            "case": case,
                            "source_sha": sources[side],
                            "model_path": models[name],
                            "runtime_pins": config["runtime_pins"],
                        },
                    )
                    print(f"Measuring {stem}", flush=True)
                    run_child(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("worker.py")),
                            "--source",
                            str(roots[side]),
                            "--spec",
                            str(spec_path.resolve()),
                            "--output",
                            str(result_path.resolve()),
                        ],
                        args.output / f"{stem}.log",
                        180,
                        env,
                    )
                    records.append(json.loads(result_path.read_text()))
                    _write_json(args.output / "observations.json", records)
        report = summarize(records, cases, profile["repeats"], sources)
        _write_json(args.output / "report.json", report)
        summary_path.write_text(markdown(report, sources))
    except Exception as error:
        _write_json(
            args.output / "failure.json",
            {"error": str(error), "completed_observations": len(records)},
        )
        raise


if __name__ == "__main__":
    main()
