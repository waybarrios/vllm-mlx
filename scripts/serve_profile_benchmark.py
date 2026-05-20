#!/usr/bin/env python3
"""Benchmark /v1/chat/completions throughput against a running server."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

DEFAULT_PROMPTS = [
    "Write a short poem about nature.",
    "Write a short poem about love.",
    "Write a short poem about technology.",
    "Write a short poem about space.",
    "Write a short poem about music.",
    "Write a short poem about art.",
    "Write a short poem about science.",
    "Write a short poem about history.",
    "Write a short poem about food.",
    "Write a short poem about travel.",
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


def _call_chat(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    disable_thinking: bool = False,
    extra_body: dict | None = None,
) -> dict:
    started = time.perf_counter()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if disable_thinking:
        body["enable_thinking"] = False
    if extra_body:
        body.update(extra_body)
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=body,
        timeout=timeout,
    )
    latency_s = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()
    usage = payload.get("usage") or {}
    return {
        "prompt": prompt,
        "latency_s": round(latency_s, 4),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _is_ready_response(response: requests.Response) -> bool:
    if not response.ok:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    if payload.get("status") == "healthy":
        return True
    if payload.get("object") == "list" and isinstance(payload.get("data"), list):
        return True
    return False


def _wait_for_health(
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
    while time.time() < deadline:
        for probe_url in probe_urls:
            try:
                response = requests.get(probe_url, timeout=2.0)
                if not _is_ready_response(response):
                    continue
                try:
                    payload = response.json()
                except ValueError:
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
            except requests.RequestException:
                pass
        time.sleep(0.5)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--prompts-file")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
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
    advertised_model_id = _wait_for_health(
        args.base_url,
        args.health_timeout,
        required_model_id_substring=args.require_model_id_substring,
    )

    # Warm the active model and server path before collecting timed results.
    if prompts and args.warmup_requests > 0:
        warmup_prompt = prompts[0]
        for _ in range(args.warmup_requests):
            _call_chat(
                base_url=args.base_url,
                model=args.model,
                prompt=warmup_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.request_timeout,
                disable_thinking=args.disable_thinking,
                extra_body=extra_body,
            )

    started = time.perf_counter()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(
                _call_chat,
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.request_timeout,
                disable_thinking=args.disable_thinking,
                extra_body=extra_body,
            )
            for prompt in prompts
        ]
        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.perf_counter() - started
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    total_tokens = sum(r["total_tokens"] for r in results)

    summary = {
        "total_time_s": round(total_time, 4),
        "prompts": len(prompts),
        "prompts_per_second": round(len(prompts) / total_time, 4),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": round(total_completion_tokens / total_time, 4),
        "throughput_tok_per_s": round(total_tokens / total_time, 4),
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "warmup_requests": args.warmup_requests,
        "model": args.model,
        "base_url": args.base_url,
        "prompts_file": args.prompts_file,
        "advertised_model_id": advertised_model_id,
        "model_id_guard_substring": args.require_model_id_substring,
        "extra_body": extra_body or None,
    }

    print("Results:")
    print(f"  Total time: {summary['total_time_s']:.2f}s")
    print(f"  Prompts: {summary['prompts']}")
    print(f"  Prompts/second: {summary['prompts_per_second']:.2f}")
    print(f"  Total prompt tokens: {summary['total_prompt_tokens']}")
    print(f"  Total completion tokens: {summary['total_completion_tokens']}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  Tokens/second: {summary['tokens_per_second']:.2f}")
    print(f"  Throughput: {summary['throughput_tok_per_s']:.2f} tok/s")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary["request_results"] = sorted(results, key=lambda item: item["prompt"])
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
