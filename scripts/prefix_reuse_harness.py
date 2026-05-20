#!/usr/bin/env python3
"""Measure repeated-prefix reuse timing against a running server."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


def _load_fixture(path: str) -> dict:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    shared_prefix = payload.get("shared_prefix")
    suffixes = payload.get("suffixes")
    if not isinstance(shared_prefix, str) or not shared_prefix.strip():
        raise ValueError(f"Fixture {fixture_path} is missing a non-empty shared_prefix")
    if not isinstance(suffixes, list) or len(suffixes) < 3:
        raise ValueError(f"Fixture {fixture_path} must include at least three suffixes")

    normalized_suffixes: list[dict[str, str]] = []
    for index, item in enumerate(suffixes[:3], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Fixture suffix #{index} must be an object")
        label = item.get("label") or f"suffix-{index}"
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Fixture suffix #{index} is missing a non-empty prompt")
        normalized_suffixes.append({"label": str(label), "prompt": prompt.strip()})

    return {
        "fixture_path": str(fixture_path),
        "shared_prefix": shared_prefix.strip(),
        "suffixes": normalized_suffixes,
    }


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
                    if not response.is_success:
                        continue
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


async def _run_stream_request(
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
    first_content_time = None
    content_chunks = 0
    collected_parts: list[str] = []

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    if disable_thinking:
        payload["enable_thinking"] = False
    if extra_body:
        payload.update(extra_body)

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
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if not content:
                    continue
                if first_content_time is None:
                    first_content_time = time.perf_counter()
                content_chunks += 1
                collected_parts.append(content)

    end_time = time.perf_counter()
    ttft_ms = (first_content_time - start_time) * 1000 if first_content_time else 0.0
    total_time_ms = (end_time - start_time) * 1000
    response_text = "".join(collected_parts)
    return {
        "ttft_ms": round(ttft_ms, 4),
        "total_time_ms": round(total_time_ms, 4),
        "content_chunk_count": content_chunks,
        "response_chars": len(response_text),
        "response_preview": response_text[:200],
    }


def _build_prompt(shared_prefix: str, suffix_prompt: str) -> str:
    return f"{shared_prefix}\n\nTask:\n{suffix_prompt}\n"


def _compute_summary(requests: list[dict]) -> dict:
    cold = requests[0]
    warm_requests = requests[1:]
    warm_ttfts = [request["ttft_ms"] for request in warm_requests]
    warm_totals = [request["total_time_ms"] for request in warm_requests]
    mean_warm_ttft = sum(warm_ttfts) / len(warm_ttfts)
    mean_warm_total = sum(warm_totals) / len(warm_totals)
    return {
        "cold_ttft_ms": cold["ttft_ms"],
        "cold_total_time_ms": cold["total_time_ms"],
        "warm_ttft_ms_mean": round(mean_warm_ttft, 4),
        "warm_total_time_ms_mean": round(mean_warm_total, 4),
        "warm_ttft_delta_from_cold_ms_mean": round(mean_warm_ttft - cold["ttft_ms"], 4),
        "warm_total_time_delta_from_cold_ms_mean": round(
            mean_warm_total - cold["total_time_ms"], 4
        ),
    }


async def _run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--health-timeout", type=float, default=120.0)
    parser.add_argument(
        "--require-model-id-substring",
        help="Fail unless at least one /v1/models id contains this substring",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Send enable_thinking=false in request body",
    )
    parser.add_argument(
        "--extra-body-json",
        help="Extra JSON object to merge into each request body",
    )
    parser.add_argument("--json-out")
    args = parser.parse_args()

    fixture = _load_fixture(args.fixture)
    extra_body = _load_extra_body(args.extra_body_json)
    advertised_model_id = await _wait_for_health(
        args.base_url,
        args.health_timeout,
        required_model_id_substring=args.require_model_id_substring,
    )

    requests_payload: list[dict] = []
    for index, suffix in enumerate(fixture["suffixes"], start=1):
        request_metrics = await _run_stream_request(
            base_url=args.base_url,
            model=args.model,
            prompt=_build_prompt(fixture["shared_prefix"], suffix["prompt"]),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            request_timeout=args.request_timeout,
            disable_thinking=args.disable_thinking,
            extra_body=extra_body,
        )
        requests_payload.append(
            {
                "request_index": index,
                "request_type": "cold" if index == 1 else "warm",
                "suffix_label": suffix["label"],
                "suffix_prompt": suffix["prompt"],
                **request_metrics,
            }
        )

    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "fixture": fixture["fixture_path"],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "shared_prefix_chars": len(fixture["shared_prefix"]),
        "shared_prefix_lines": len(fixture["shared_prefix"].splitlines()),
        "request_count": len(requests_payload),
        "advertised_model_id": advertised_model_id,
        "model_id_guard_substring": args.require_model_id_substring,
        "extra_body": extra_body or None,
        **_compute_summary(requests_payload),
    }

    print("Prefix Reuse Results:")
    print(f"  Advertised model id: {summary['advertised_model_id']}")
    print(f"  Cold TTFT: {summary['cold_ttft_ms']:.2f} ms")
    print(f"  Warm TTFT mean: {summary['warm_ttft_ms_mean']:.2f} ms")
    print(
        "  Warm TTFT delta from cold mean: "
        f"{summary['warm_ttft_delta_from_cold_ms_mean']:.2f} ms"
    )
    print(f"  Cold total time: {summary['cold_total_time_ms']:.2f} ms")
    print(f"  Warm total time mean: {summary['warm_total_time_ms_mean']:.2f} ms")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "requests": requests_payload}
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {out_path}")

    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
