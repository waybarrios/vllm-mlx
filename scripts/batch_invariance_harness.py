#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Batch invariance harness for vllm-mlx.

Compares deterministic outputs for the same prompt set under:
1) isolated requests (one-by-one)
2) concurrent requests (batch-composition pressure proxy)

Usage:
  python scripts/batch_invariance_harness.py \
    --base-url http://localhost:8000 \
    --model mlx-community/Qwen3-4B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

DEFAULT_PROMPTS = [
    "Write one sentence explaining why caching helps inference throughput.",
    "List three steps to verify an API endpoint is healthy.",
    "What is the capital of France? Answer in one word.",
    "Return exactly: ping",
    "Name two risks of running without authentication.",
    "Summarize why deterministic decoding matters in testing.",
    "What does a 503 status code usually indicate?",
    "Give a one-line definition of batch invariance.",
    "Provide two bullet points on memory pressure guardrails.",
    "Output the word READY and nothing else.",
]

Z_SCORE_BY_CONFIDENCE = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}


class HarnessRequestError(RuntimeError):
    """Raised when probe requests fail after retry."""


@dataclass
class ProbeResult:
    prompt: str
    output: str
    latency_s: float
    finish_reason: str | None


def _load_prompts(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_PROMPTS)
    prompt_path = Path(path)
    lines = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in lines if line]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _run_probe(
    base_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    timeout_s: float,
    api_key: str | None,
    retries: int,
    retry_backoff_s: float,
) -> ProbeResult:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    attempts = max(1, retries + 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            t0 = time.perf_counter()
            response = requests.post(
                url,
                headers=_headers(api_key),
                json=payload,
                timeout=timeout_s,
            )
            latency_s = time.perf_counter() - t0
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            message = choice.get("message", {})
            return ProbeResult(
                prompt=prompt,
                output=message.get("content") or "",
                latency_s=latency_s,
                finish_reason=choice.get("finish_reason"),
            )
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            last_error = exc
            if attempt >= attempts:
                break
            sleep_s = retry_backoff_s * attempt
            if sleep_s > 0:
                time.sleep(sleep_s)
    raise HarnessRequestError(
        f"Request failed after {attempts} attempt(s) for prompt: {prompt!r}. "
        f"Last error: {last_error}"
    )


def _tokenize(text: str) -> list[str]:
    return text.split()


def _token_agreement(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    max_len = max(len(ta), len(tb), 1)
    matches = 0
    for i in range(max_len):
        tok_a = ta[i] if i < len(ta) else None
        tok_b = tb[i] if i < len(tb) else None
        if tok_a == tok_b:
            matches += 1
    return matches / max_len


def _run_serial(
    prompts: list[str],
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout_s: float,
    api_key: str | None,
    retries: int,
    retry_backoff_s: float,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for prompt in prompts:
        results.append(
            _run_probe(
                base_url,
                model,
                prompt,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                api_key=api_key,
                retries=retries,
                retry_backoff_s=retry_backoff_s,
            )
        )
    return results


def _run_concurrent(
    prompts: list[str],
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout_s: float,
    api_key: str | None,
    concurrency: int,
    retries: int,
    retry_backoff_s: float,
) -> list[ProbeResult]:
    results: list[ProbeResult] = [None] * len(prompts)  # type: ignore[assignment]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures: dict[concurrent.futures.Future[ProbeResult], int] = {}
        for idx, prompt in enumerate(prompts):
            fut = pool.submit(
                _run_probe,
                base_url,
                model,
                prompt,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                api_key=api_key,
                retries=retries,
                retry_backoff_s=retry_backoff_s,
            )
            futures[fut] = idx
        errors: list[str] = []
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                errors.append(f"[prompt_index={idx}] {exc}")
        if errors:
            head = "\n".join(errors[:3])
            tail_note = ""
            if len(errors) > 3:
                tail_note = f"\n... and {len(errors) - 3} more error(s)"
            raise HarnessRequestError(
                "Concurrent probe run failed. This usually means the server is down, "
                "restarting, or overloaded under concurrency.\n"
                f"{head}{tail_note}"
            )
    return results


def _compute_confidence_interval(
    values: list[float],
    *,
    confidence: float,
) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot compute confidence interval for empty values")
    mean = statistics.mean(values)
    if len(values) <= 1:
        return {
            "mean": mean,
            "stdev": 0.0,
            "ci_lower": mean,
            "ci_upper": mean,
        }
    stdev = statistics.stdev(values)
    z_score = Z_SCORE_BY_CONFIDENCE[confidence]
    half_width = z_score * (stdev / math.sqrt(len(values)))
    return {
        "mean": mean,
        "stdev": stdev,
        "ci_lower": max(0.0, mean - half_width),
        "ci_upper": min(1.0, mean + half_width),
    }


def _run_single_pass(
    prompts: list[str],
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout_s: float,
    api_key: str | None,
    concurrency: int,
    run_index: int,
    retries: int,
    retry_backoff_s: float,
) -> dict[str, Any]:
    serial = _run_serial(
        prompts,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        api_key=api_key,
        retries=retries,
        retry_backoff_s=retry_backoff_s,
    )
    concurrent = _run_concurrent(
        prompts,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        api_key=api_key,
        concurrency=concurrency,
        retries=retries,
        retry_backoff_s=retry_backoff_s,
    )

    exact_matches = 0
    per_prompt_agreement: list[float] = []
    rows: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        a = serial[idx]
        b = concurrent[idx]
        agreement = _token_agreement(a.output, b.output)
        exact = a.output == b.output
        if exact:
            exact_matches += 1
        per_prompt_agreement.append(agreement)
        rows.append(
            {
                "index": idx,
                "prompt": prompt,
                "serial_output": a.output,
                "concurrent_output": b.output,
                "token_agreement": round(agreement, 4),
                "exact_match": exact,
                "serial_latency_s": round(a.latency_s, 4),
                "concurrent_latency_s": round(b.latency_s, 4),
            }
        )

    exact_match_rate = exact_matches / len(prompts)
    token_agreement_rate = statistics.mean(per_prompt_agreement)
    return {
        "run_index": run_index,
        "exact_match_rate": exact_match_rate,
        "token_agreement_rate": token_agreement_rate,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch invariance harness")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Server base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier to include in request payload",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Optional text file (one prompt per line)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens per probe request",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout seconds",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrent request count for batch-composition run",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key (Authorization: Bearer)",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write JSON report",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated serial+concurrent passes for confidence intervals",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        choices=sorted(Z_SCORE_BY_CONFIDENCE),
        default=0.95,
        help="Confidence level for interval output",
    )
    parser.add_argument(
        "--run-cooldown",
        type=float,
        default=0.0,
        help="Optional seconds to sleep between repeated runs",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=2,
        help="Number of retries per failed probe request",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=0.5,
        help="Backoff base seconds between retries",
    )
    args = parser.parse_args()

    prompts = _load_prompts(args.prompts_file)
    concurrency = max(1, min(args.concurrency, len(prompts)))
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.run_cooldown < 0:
        raise ValueError("--run-cooldown must be >= 0")
    if args.request_retries < 0:
        raise ValueError("--request-retries must be >= 0")
    if args.retry_backoff < 0:
        raise ValueError("--retry-backoff must be >= 0")

    run_reports: list[dict[str, Any]] = []
    for run_index in range(args.runs):
        print(f"Run {run_index + 1}/{args.runs}: serial pass on {len(prompts)} prompts...")
        print(f"Run {run_index + 1}/{args.runs}: concurrent pass (concurrency={concurrency})...")
        try:
            run_report = _run_single_pass(
                prompts,
                base_url=args.base_url,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
                api_key=args.api_key,
                concurrency=concurrency,
                run_index=run_index,
                retries=args.request_retries,
                retry_backoff_s=args.retry_backoff,
            )
        except HarnessRequestError as exc:
            raise SystemExit(
                "Harness failed due to request/connection errors.\n"
                f"{exc}\n\n"
                "Next checks:\n"
                "1) Confirm server is still listening on --base-url.\n"
                "2) Check server logs for OOM/restart/crash during concurrent pass.\n"
                "3) Retry with lower concurrency (e.g., --concurrency 4).\n"
                "4) Optionally reduce --max-tokens for stability."
            ) from exc
        run_reports.append(run_report)
        print(
            f"Run {run_index + 1}/{args.runs}: "
            f"token={run_report['token_agreement_rate'] * 100:.2f}% "
            f"exact={run_report['exact_match_rate'] * 100:.2f}%"
        )
        if args.run_cooldown > 0 and (run_index + 1) < args.runs:
            time.sleep(args.run_cooldown)

    token_values = [r["token_agreement_rate"] for r in run_reports]
    exact_values = [r["exact_match_rate"] for r in run_reports]
    token_summary = _compute_confidence_interval(token_values, confidence=args.confidence)
    exact_summary = _compute_confidence_interval(exact_values, confidence=args.confidence)
    ci_percent = int(args.confidence * 100)
    latest_rows = run_reports[-1]["rows"]

    print()
    print("Batch Invariance Report")
    print("-" * 60)
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Runs: {args.runs}")
    print(f"Exact output match rate: {exact_summary['mean'] * 100:.2f}%")
    print(f"Token agreement rate:    {token_summary['mean'] * 100:.2f}%")
    print(
        f"{ci_percent}% CI token agreement: "
        f"[{token_summary['ci_lower'] * 100:.2f}%, {token_summary['ci_upper'] * 100:.2f}%]"
    )
    print(
        f"{ci_percent}% CI exact match:    "
        f"[{exact_summary['ci_lower'] * 100:.2f}%, {exact_summary['ci_upper'] * 100:.2f}%]"
    )
    print("-" * 60)
    for row in latest_rows:
        print(
            f"[{row['index']:02d}] agreement={row['token_agreement']:.2f} "
            f"exact={row['exact_match']}"
        )

    if token_summary["ci_lower"] < 0.95:
        print("Result: potential batch invariance violation (<95% token agreement).")
    else:
        print("Result: no significant batch invariance violation detected.")

    report = {
        "model": args.model,
        "base_url": args.base_url,
        "prompt_count": len(prompts),
        "max_tokens": args.max_tokens,
        "concurrency": concurrency,
        "runs": args.runs,
        "confidence_level": args.confidence,
        "exact_match_rate": exact_summary["mean"],
        "token_agreement_rate": token_summary["mean"],
        "exact_match_summary": exact_summary,
        "token_agreement_summary": token_summary,
        "rows": latest_rows,
        "run_reports": run_reports,
        "timestamp_epoch": int(time.time()),
    }
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote JSON report: {out_path}")


if __name__ == "__main__":
    main()
