#!/usr/bin/env python3
"""Measure streaming latency metrics against a running server."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
]


def _load_prompts(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_PROMPTS)
    prompt_path = Path(path)
    lines = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in lines if line]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts


def _load_extra_body(value: str | None) -> dict:
    if not value:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--extra-body-json must decode to a JSON object")
    return payload


def _build_request_payload(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    disable_thinking: bool,
    extra_body: dict | None,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if disable_thinking:
        payload["enable_thinking"] = False
    if not extra_body:
        return payload

    merged_payload = dict(payload)
    extra_stream_options = extra_body.get("stream_options")
    if extra_stream_options is not None and not isinstance(extra_stream_options, dict):
        raise ValueError("--extra-body-json stream_options must decode to a JSON object")

    for key, value in extra_body.items():
        if key == "stream_options":
            continue
        merged_payload[key] = value

    stream_options = dict(payload["stream_options"])
    if isinstance(extra_stream_options, dict):
        stream_options.update(extra_stream_options)
    stream_options["include_usage"] = True
    merged_payload["stream_options"] = stream_options
    return merged_payload


def _extract_model_ids(payload: dict) -> list[str]:
    if payload.get("object") != "list" or not isinstance(payload.get("data"), list):
        return []
    model_ids: list[str] = []
    for item in payload["data"]:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id:
            model_ids.append(model_id)
    return model_ids


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "stdev": 0.0,
        }
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "stdev": round(stdev, 4),
    }


def _is_ready_payload(payload: dict) -> bool:
    if payload.get("status") == "healthy":
        return True
    if payload.get("object") == "list" and isinstance(payload.get("data"), list):
        return True
    return False


async def _wait_for_health(
    base_url: str,
    timeout_s: float,
    required_model_id_substring: str | None = None,
) -> str:
    deadline = time.time() + timeout_s
    probe_urls = [
        f"{base_url.rstrip('/')}/health",
        f"{base_url.rstrip('/')}/v1/models",
    ]
    last_model_ids: list[str] = []
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.time() < deadline:
            for probe_url in probe_urls:
                try:
                    response = await client.get(probe_url)
                    if response.is_success:
                        payload = response.json()
                        if not _is_ready_payload(payload):
                            continue
                        model_ids = _extract_model_ids(payload)
                        if model_ids:
                            last_model_ids = model_ids
                            if required_model_id_substring:
                                for model_id in model_ids:
                                    if required_model_id_substring in model_id:
                                        return model_id
                            else:
                                return model_ids[0]
                except (httpx.HTTPError, ValueError):
                    pass
            await asyncio.sleep(0.5)
    if required_model_id_substring:
        raise TimeoutError(
            "Server health/model-id check did not pass within "
            f"{timeout_s:.1f}s; required substring="
            f"{required_model_id_substring!r}, last advertised model ids={last_model_ids!r}"
        )
    raise TimeoutError(
        "Server health check did not pass within "
        f"{timeout_s:.1f}s; last advertised model ids={last_model_ids!r}"
    )


async def _measure_streaming_latency(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout: float,
    disable_thinking: bool = False,
    extra_body: dict | None = None,
) -> dict:
    start_time = time.perf_counter()
    first_token_time = None
    last_content_time = None
    content_chunk_count = 0
    usage = {}
    payload = _build_request_payload(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        disable_thinking=disable_thinking,
        extra_body=extra_body,
    )

    async with httpx.AsyncClient(timeout=request_timeout) as client:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                current_time = time.perf_counter()
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                chunk_usage = chunk.get("usage")
                if isinstance(chunk_usage, dict):
                    usage = chunk_usage
                choices = chunk.get("choices")
                delta = {}
                if isinstance(choices, list) and choices:
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        delta = first_choice.get("delta", {})
                content = delta.get("content", "") if isinstance(delta, dict) else ""
                if content:
                    content_chunk_count += 1
                    if first_token_time is None:
                        first_token_time = current_time
                    last_content_time = current_time

    end_time = time.perf_counter()
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)

    if first_token_time is not None and completion_tokens <= 0:
        raise RuntimeError(
            "Streaming response did not include usage.completion_tokens. "
            "Benchmark results would be chunk-count confounded."
        )

    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0.0
    total_time_ms = (end_time - start_time) * 1000
    emission_window_ms = (
        (last_content_time - first_token_time) * 1000
        if first_token_time is not None and last_content_time is not None
        else 0.0
    )
    mean_itl_ms = (
        emission_window_ms / (completion_tokens - 1)
        if completion_tokens > 1
        else 0.0
    )

    return {
        "prompt": prompt,
        "ttft_ms": round(ttft_ms, 4),
        "total_time_ms": round(total_time_ms, 4),
        "token_count": completion_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "content_chunk_count": content_chunk_count,
        "emission_window_ms": round(emission_window_ms, 4),
        "mean_itl_ms": round(mean_itl_ms, 4),
    }


async def _run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-file")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--health-timeout", type=float, default=120.0)
    parser.add_argument(
        "--require-model-id-substring",
        help="Fail unless at least one /v1/models id contains this substring",
    )
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Send enable_thinking=false in request body")
    parser.add_argument(
        "--extra-body-json",
        help="Extra JSON object to merge into each request body",
    )
    parser.add_argument("--json-out")
    args = parser.parse_args()

    prompts = _load_prompts(args.prompts_file)
    extra_body = _load_extra_body(args.extra_body_json)
    advertised_model_id = await _wait_for_health(
        args.base_url,
        args.health_timeout,
        required_model_id_substring=args.require_model_id_substring,
    )

    if prompts and args.warmup_requests > 0:
        warmup_prompt = prompts[0]
        for _ in range(args.warmup_requests):
            await _measure_streaming_latency(
                base_url=args.base_url,
                model=args.model,
                prompt=warmup_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                request_timeout=args.request_timeout,
                disable_thinking=args.disable_thinking,
                extra_body=extra_body,
            )

    runs: list[dict] = []
    for prompt in prompts:
        for _ in range(args.iterations):
            runs.append(
                await _measure_streaming_latency(
                    base_url=args.base_url,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    request_timeout=args.request_timeout,
                    disable_thinking=args.disable_thinking,
                    extra_body=extra_body,
                )
            )

    ttft_values = [run["ttft_ms"] for run in runs]
    total_values = [run["total_time_ms"] for run in runs]
    token_values = [run["completion_tokens"] for run in runs]
    itl_values = [run["mean_itl_ms"] for run in runs if run["completion_tokens"] > 1]
    chunk_count_values = [run["content_chunk_count"] for run in runs]
    total_time_s = sum(total_values) / 1000.0
    total_tokens = sum(token_values)
    throughput = (total_tokens / total_time_s) if total_time_s > 0 else 0.0

    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "prompts_file": args.prompts_file,
        "iterations": args.iterations,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "prompt_count": len(prompts),
        "run_count": len(runs),
        "warmup_requests": args.warmup_requests,
        "metric_method": "stream_usage_completion_tokens",
        "advertised_model_id": advertised_model_id,
        "model_id_guard_substring": args.require_model_id_substring,
        "extra_body": extra_body or None,
        "ttft_ms": _stats(ttft_values),
        "itl_ms": _stats(itl_values),
        "total_time_ms": _stats(total_values),
        "content_chunk_count": _stats(chunk_count_values),
        "tokens_per_second": round(throughput, 4),
        "total_tokens": total_tokens,
    }

    print("Streaming Latency Results:")
    print(f"  Runs: {summary['run_count']}")
    print(f"  Advertised model id: {summary['advertised_model_id']}")
    print(f"  TTFT mean: {summary['ttft_ms']['mean']:.2f} ms")
    print(f"  ITL mean: {summary['itl_ms']['mean']:.2f} ms")
    print(f"  Total time mean: {summary['total_time_ms']['mean']:.2f} ms")
    print(f"  Tokens/sec: {summary['tokens_per_second']:.2f}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "runs": runs}
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {out_path}")

    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
