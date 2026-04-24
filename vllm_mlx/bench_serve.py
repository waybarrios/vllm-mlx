# SPDX-License-Identifier: Apache-2.0
"""
Serving benchmark for vllm-mlx.

Measures end-to-end HTTP performance of a running vllm-mlx server:
- Time to First Token (TTFT)
- Time Per Output Token (TPOT)
- End-to-end latency
- Generation and prompt throughput
- Concurrent request handling
- KV cache hit rates
- Metal memory utilization

This module has no MLX dependency and can be imported on any platform.
It is a pure HTTP client that talks to a running OpenAI-compatible server.
"""

import asyncio
import csv as csv_mod
import dataclasses as _dataclasses
import io
import itertools
import json
import logging
import math
import platform
import re
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from tabulate import tabulate as _tabulate

# ---------------------------------------------------------------------------
# Prompt set loading
# ---------------------------------------------------------------------------

_BUILTIN_DIR = Path(__file__).parent / "bench_serve_prompts"
_BUILTIN_NAMES = {"short", "medium", "long", "thinking"}


@dataclass
class WorkloadCase:
    """One declarative benchmark case for contract-style serving tests."""

    case_id: str
    messages: list[dict]
    request_path: Optional[str] = None
    max_tokens: Optional[int] = None
    enable_thinking: Optional[bool] = None
    extra_body: Optional[dict] = None
    policy_timeout_ms: Optional[int] = None
    checks: Optional[dict] = None
    tags: tuple[str, ...] = ()


@dataclass
class Workload:
    """Normalized bench-serve workload manifest."""

    name: str
    description: str
    defaults: dict
    cases: list[WorkloadCase]


def load_prompt_set(name_or_path: str) -> list[list[dict]]:
    """Load a prompt set by builtin name or file path.

    Builtin sets (``short``, ``medium``, ``long``, ``thinking``) are loaded
    from the ``bench_serve_prompts/`` directory next to this module.  Any
    other value is treated as a filesystem path and loaded directly.

    Two file formats are accepted (detected automatically):

    1. **Flat** — list of single message dicts. Each dict becomes a
       single-message prompt. Backwards-compatible with the original format.

       ``[{"role": "user", "content": "..."}, ...]``

    2. **Multi-message** — list of message-dict lists. Each inner list is a
       full chat history (e.g. ``[system, user]``). Use this format when you
       want to benchmark with system prompts that match an ``--warm-prompts``
       warm-up, or to simulate multi-turn conversation.

       ``[[{"role":"system","content":"..."}, {"role":"user","content":"..."}], ...]``

    Returns:
        A list of message-dict lists, i.e. every entry is a full chat history.
        Flat-format files are normalized to single-element lists.

    Raises:
        FileNotFoundError: If ``name_or_path`` is not a known builtin name and
            the path does not exist, or if a builtin name is requested but its
            JSON file is missing from the package.
        ValueError: If the file shape is not recognised.
    """
    if name_or_path in _BUILTIN_NAMES:
        target = _BUILTIN_DIR / f"{name_or_path}.json"
        if not target.exists():
            raise FileNotFoundError(
                f"Builtin prompt set '{name_or_path}' not found at {target}"
            )
        with target.open() as fh:
            raw = json.load(fh)
    else:
        path = Path(name_or_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Unknown prompt set name or missing file: '{name_or_path}'. "
                f"Builtin names are: {sorted(_BUILTIN_NAMES)}"
            )
        with path.open() as fh:
            raw = json.load(fh)

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Prompt file must be a non-empty JSON list: {name_or_path}")

    # Auto-detect format: dict entries = flat; list entries = multi-message.
    first = raw[0]
    if isinstance(first, dict):
        # Flat format: wrap each message in a single-element list.
        return [[msg] for msg in raw]
    if isinstance(first, list):
        return raw
    raise ValueError(
        f"Prompt entries must be dict or list, got {type(first).__name__} "
        f"in {name_or_path}"
    )


def _require_message_list(value: Any, *, label: str) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}: messages must be a non-empty list")
    for idx, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"{label}: message {idx} must be an object")
        if "role" not in message or "content" not in message:
            raise ValueError(f"{label}: message {idx} must include role and content")
    return value


def _load_case_request(path: str, *, workload_path: Path, case_id: str) -> dict:
    request_path = Path(path).expanduser()
    if not request_path.is_absolute():
        request_path = workload_path.parent / request_path
    with request_path.open() as fh:
        request = json.load(fh)
    if not isinstance(request, dict):
        raise ValueError(f"{case_id}: request_path must point to a JSON object")
    return request


def _request_extra_body(request: dict) -> dict:
    reserved = {
        "model",
        "messages",
        "max_tokens",
        "stream",
        "stream_options",
        "enable_thinking",
    }
    return {key: value for key, value in request.items() if key not in reserved}


def load_workload(path: str | Path) -> Workload:
    """Load a declarative serving benchmark workload.

    Workloads are for product-like qualification where each case can carry
    request settings, comparison-only policy timeouts, and quality checks.
    Timeout fields are metadata unless the runner explicitly uses them as a
    transport limit; they are not treated as hardware capability claims.
    """
    workload_path = Path(path).expanduser()
    with workload_path.open() as fh:
        raw = json.load(fh)

    if not isinstance(raw, dict):
        raise ValueError("workload root must be a JSON object")
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("workload must contain a non-empty cases list")

    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("workload defaults must be an object")

    cases: list[WorkloadCase] = []
    for idx, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError(f"case {idx}: case must be an object")
        case_id = str(item.get("id") or f"case_{idx + 1}")
        request_path = item.get("request_path")
        request_defaults: dict = {}
        if request_path is not None:
            request_defaults = _load_case_request(
                str(request_path), workload_path=workload_path, case_id=case_id
            )
        messages = _require_message_list(
            item.get("messages", request_defaults.get("messages")),
            label=case_id,
        )
        extra_body = item.get("extra_body", defaults.get("extra_body"))
        request_extra = _request_extra_body(request_defaults)
        if extra_body:
            request_extra.update(extra_body)
        extra_body = request_extra or None
        if extra_body is not None and not isinstance(extra_body, dict):
            raise ValueError(f"{case_id}: extra_body must be an object")
        checks = item.get("checks", defaults.get("checks"))
        if checks is not None and not isinstance(checks, dict):
            raise ValueError(f"{case_id}: checks must be an object")
        tags = item.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, list):
            raise ValueError(f"{case_id}: tags must be a list or string")

        cases.append(
            WorkloadCase(
                case_id=case_id,
                messages=messages,
                request_path=str(request_path) if request_path is not None else None,
                max_tokens=item.get(
                    "max_tokens",
                    request_defaults.get("max_tokens", defaults.get("max_tokens")),
                ),
                enable_thinking=item.get(
                    "enable_thinking",
                    request_defaults.get(
                        "enable_thinking", defaults.get("enable_thinking")
                    ),
                ),
                extra_body=extra_body,
                policy_timeout_ms=item.get(
                    "policy_timeout_ms", defaults.get("policy_timeout_ms")
                ),
                checks=checks,
                tags=tuple(str(tag) for tag in tags),
            )
        )

    return Workload(
        name=str(raw.get("name") or workload_path.stem),
        description=str(raw.get("description") or ""),
        defaults=defaults,
        cases=cases,
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchServeResult:
    """Aggregated results from a single bench-serve run configuration."""

    # --- Identity ---
    run_id: str = ""
    timestamp: str = ""
    tag: str = ""

    # --- Hardware ---
    chip: str = ""
    gpu_cores: int = 0
    memory_gb: float = 0.0
    bandwidth_gbs: float = 0.0
    os_version: str = ""

    # --- Runtime ---
    model_id: str = ""
    model_type: str = ""
    engine_type: str = ""
    mtp_enabled: bool = False
    specprefill: bool = False
    kv_quant: str = ""
    cache_type: str = ""

    # --- Config ---
    prompt_set: str = ""
    concurrency: int = 1
    max_tokens: int = 256
    enable_thinking: Optional[bool] = None
    extra_body: str = ""
    repetition: int = 0
    prompt_tokens: int = 0

    # --- Latency (milliseconds) ---
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_latency_ms: float = 0.0

    # --- Throughput ---
    gen_tps: float = 0.0
    prompt_tps: float = 0.0
    throughput_tps: float = 0.0
    requests_per_s: float = 0.0

    # --- Memory (gigabytes) ---
    metal_active_gb: float = 0.0
    metal_peak_gb: float = 0.0
    metal_cache_gb: float = 0.0

    # --- Cache ---
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    tokens_saved: int = 0

    # --- Validation ---
    validated: bool = True


# ---------------------------------------------------------------------------
# Sweep configuration type alias and combinatorial expansion
# ---------------------------------------------------------------------------

# (prompt_set, concurrency, thinking, extra_body, repetition_index)
SweepConfig = tuple[str, int, Optional[bool], str, int]


def expand_sweep(
    prompt_sets: list[str],
    concurrencies: list[int],
    thinking_values: list[Optional[bool]],
    extra_bodies: list[str],
    repetitions: int,
) -> list[SweepConfig]:
    """Expand sweep parameters into a flat list of configurations.

    Performs the full Cartesian product of all input dimensions and then
    unfolds each combination across ``repetitions`` repetition indices
    (0-based).

    Args:
        prompt_sets: Names or paths of prompt sets to include.
        concurrencies: Concurrency levels to test (e.g. ``[1, 4, 16]``).
        thinking_values: Values for ``enable_thinking`` (e.g.
            ``[None, True, False]``).
        extra_bodies: JSON strings (or empty string) to pass as extra body
            parameters on each request.
        repetitions: Number of times to repeat each unique combination.
            Each repeat gets a distinct 0-based repetition index.

    Returns:
        A list of :data:`SweepConfig` tuples in the order::

            (prompt_set, concurrency, thinking, extra_body, repetition_index)
    """
    configs: list[SweepConfig] = []
    for prompt_set, concurrency, thinking, extra_body, rep in itertools.product(
        prompt_sets, concurrencies, thinking_values, extra_bodies, range(repetitions)
    ):
        configs.append((prompt_set, concurrency, thinking, extra_body, rep))
    return configs


# ---------------------------------------------------------------------------
# Task 3: Server auto-detection + hardware fingerprint
# ---------------------------------------------------------------------------


def parse_health_response(data: dict) -> dict:
    """Extract model identity fields from a GET /health response.

    Args:
        data: Parsed JSON body from the /health endpoint.  Expected shape::

            {"status": "healthy", "model_loaded": True,
             "model_name": "...", "model_type": "llm"|"mllm"}

    Returns:
        ``{"model_name": str, "model_type": str}``
    """
    return {
        "model_name": data.get("model_name", ""),
        "model_type": data.get("model_type", ""),
    }


def parse_status_response(data: dict) -> dict:
    """Extract metal and cache info from a GET /v1/status response.

    Args:
        data: Parsed JSON body from the /v1/status endpoint.  Metal info is
            expected under ``data["metal"]`` and cache info under
            ``data["cache"]``.  Missing keys are handled gracefully.

    Returns:
        ``{"model": str, "metal_active_gb": float, "metal_peak_gb": float,
        "metal_cache_gb": float, "cache_type": str}``
    """
    metal = data.get("metal") or {}
    cache = data.get("cache") or {}
    return {
        "model": data.get("model", ""),
        "metal_active_gb": float(metal.get("active_gb") or 0.0),
        "metal_peak_gb": float(metal.get("peak_gb") or 0.0),
        "metal_cache_gb": float(metal.get("cache_gb") or 0.0),
        "cache_type": cache.get("type", "") or "",
    }


def parse_metrics_text(text: str) -> dict:
    """Parse Prometheus text exposition format from GET /metrics.

    Extracts the three prefix-cache counters used for bench reporting.

    Args:
        text: Raw response body from the /metrics endpoint.

    Returns:
        ``{"cache_hits": int, "cache_misses": int, "tokens_saved": int}``
        — each value defaults to ``0`` when the metric line is absent.
    """

    def _extract(metric_name: str) -> int:
        pattern = rf"^{re.escape(metric_name)}\s+(\d+)"
        m = re.search(pattern, text, re.MULTILINE)
        return int(m.group(1)) if m else 0

    return {
        "cache_hits": _extract("vllm_prefix_cache_hits_total"),
        "cache_misses": _extract("vllm_prefix_cache_misses_total"),
        "tokens_saved": _extract("vllm_prefix_cache_tokens_saved_total"),
    }


def detect_hardware_fingerprint() -> dict:
    """Return a hardware fingerprint dict for the current machine.

    Tries to import :func:`vllm_mlx.optimizations.detect_hardware` (which
    requires MLX).  Falls back to reading ``hw.memsize`` via ``sysctl`` when
    MLX is unavailable.  ``os_version`` is always obtained from
    :func:`platform.platform`.

    Returns:
        ``{"chip": str, "gpu_cores": int, "memory_gb": float,
        "bandwidth_gbs": float, "os_version": str}``
    """
    os_version = platform.platform()

    try:
        from .optimizations import detect_hardware  # type: ignore[import]

        hw = detect_hardware()
        return {
            "chip": hw.chip_name,
            "gpu_cores": hw.gpu_cores,
            "memory_gb": hw.total_memory_gb,
            "bandwidth_gbs": hw.memory_bandwidth_gbs,
            "os_version": os_version,
        }
    except Exception:
        pass

    # Fallback: use sysctl for memory, leave chip/cores/bandwidth unknown.
    memory_gb = 0.0
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        memory_gb = int(result.stdout.strip()) / (1024**3)
    except Exception:
        pass

    return {
        "chip": "",
        "gpu_cores": 0,
        "memory_gb": memory_gb,
        "bandwidth_gbs": 0.0,
        "os_version": os_version,
    }


async def auto_detect_runtime(client: httpx.AsyncClient, base_url: str) -> dict:
    """Query the running server and return a runtime descriptor dict.

    Hits ``/health``, ``/v1/models``, and ``/v1/status`` in sequence.
    Each call is wrapped in an :exc:`httpx.HTTPError` guard so a missing
    endpoint does not abort the whole detection.

    Args:
        client: An open :class:`httpx.AsyncClient`.
        base_url: Base URL of the server (e.g. ``"http://localhost:8080"``).

    Returns:
        Dict with keys: ``model_id``, ``model_type``, ``engine_type``,
        ``mtp_enabled``, ``specprefill``, ``kv_quant``, ``cache_type``,
        ``metal_active_gb``, ``metal_peak_gb``, ``metal_cache_gb``.
    """
    result: dict = {
        "model_id": "",
        "model_type": "",
        "engine_type": "",
        "mtp_enabled": False,
        "specprefill": False,
        "kv_quant": "",
        "cache_type": "",
        "metal_active_gb": 0.0,
        "metal_peak_gb": 0.0,
        "metal_cache_gb": 0.0,
    }

    # /health
    try:
        resp = await client.get(f"{base_url}/health")
        resp.raise_for_status()
        health = parse_health_response(resp.json())
        result["model_type"] = health.get("model_type", "")
    except httpx.HTTPError:
        pass

    # /v1/models
    try:
        resp = await client.get(f"{base_url}/v1/models")
        resp.raise_for_status()
        models_data = resp.json()
        models = models_data.get("data") or []
        if models:
            result["model_id"] = models[0].get("id", "")
    except httpx.HTTPError:
        pass

    # /v1/status
    try:
        resp = await client.get(f"{base_url}/v1/status")
        resp.raise_for_status()
        status = parse_status_response(resp.json())
        result["cache_type"] = status.get("cache_type", "")
        result["metal_active_gb"] = status.get("metal_active_gb", 0.0)
        result["metal_peak_gb"] = status.get("metal_peak_gb", 0.0)
        result["metal_cache_gb"] = status.get("metal_cache_gb", 0.0)
        raw = resp.json()
        result["engine_type"] = raw.get("engine_type", "")
        result["mtp_enabled"] = bool(raw.get("mtp_enabled", False))
        result["specprefill"] = bool(raw.get("specprefill", False))
        result["kv_quant"] = raw.get("kv_quant", "") or ""
    except httpx.HTTPError:
        pass

    return result


async def scrape_metrics(client: httpx.AsyncClient, base_url: str) -> dict:
    """Scrape Prometheus metrics from the server.

    Args:
        client: An open :class:`httpx.AsyncClient`.
        base_url: Base URL of the server.

    Returns:
        Parsed metrics dict (see :func:`parse_metrics_text`), or an empty
        dict if the endpoint is unreachable.
    """
    try:
        resp = await client.get(f"{base_url}/metrics")
        resp.raise_for_status()
        return parse_metrics_text(resp.text)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Task 4: SSE streaming core + token counting + request timing
# ---------------------------------------------------------------------------


def parse_sse_line(line: str) -> Optional[dict]:
    """Parse one Server-Sent Events line from a streaming chat completion.

    Args:
        line: A single raw line from the SSE stream (may or may not include
            a trailing newline — it is stripped before processing).

    Returns:
        ``None`` for blank lines, comment lines (starting with ``:``) and the
        ``data: [DONE]`` sentinel.  For all other ``data:`` lines the JSON is
        parsed and a dict is returned::

            {"content": str, "finish_reason": Optional[str], "usage": Optional[dict]}

        Missing keys (``choices``, ``delta``, ``content``) are handled
        gracefully and default to empty string / ``None``.
    """
    line = line.strip()
    if not line:
        return None
    if line.startswith(":"):
        return None
    if line == "data: [DONE]":
        return None
    if not line.startswith("data: "):
        return None

    payload = line[len("data: ") :]
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return None

    choices = chunk.get("choices") or []
    delta = choices[0].get("delta", {}) if choices else {}
    content = delta.get("content", "") or ""
    finish_reason = choices[0].get("finish_reason") if choices else None
    usage = chunk.get("usage")

    return {
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
    }


def compute_request_metrics(
    t_start: float,
    t_first_token: float,
    token_times: list,
    t_end: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    """Compute standard latency and throughput metrics for a single request.

    All time arguments are :func:`time.perf_counter` values (seconds as
    floats).

    Args:
        t_start: Timestamp immediately before the request was sent.
        t_first_token: Timestamp when the first content token was received.
        token_times: List of timestamps, one per content token (including the
            first).  When there is only one token ``tpot_ms`` is ``0.0``.
        t_end: Timestamp after the final SSE chunk was consumed.
        prompt_tokens: Number of prompt tokens reported by the server.
        completion_tokens: Number of completion tokens generated.

    Returns:
        Dict with keys ``ttft_ms``, ``tpot_ms``, ``e2e_latency_ms``,
        ``gen_tps``, ``prompt_tps`` — all floats.
    """
    ttft_ms = (t_first_token - t_start) * 1000.0
    e2e_latency_ms = (t_end - t_start) * 1000.0

    # TPOT: mean inter-token gap across all generated tokens.
    if len(token_times) > 1:
        intervals = [
            token_times[i] - token_times[i - 1] for i in range(1, len(token_times))
        ]
        tpot_ms = statistics.mean(intervals) * 1000.0
    else:
        tpot_ms = 0.0

    # Use last token time (not t_end which includes HTTP teardown)
    t_last_token = token_times[-1] if token_times else t_end
    gen_duration = t_last_token - t_first_token
    gen_tps = completion_tokens / gen_duration if gen_duration > 0 else 0.0

    prompt_duration = t_first_token - t_start
    prompt_tps = prompt_tokens / prompt_duration if prompt_duration > 0 else 0.0

    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_latency_ms": e2e_latency_ms,
        "gen_tps": gen_tps,
        "prompt_tps": prompt_tps,
    }


async def count_prompt_tokens(
    client: httpx.AsyncClient,
    base_url: str,
    messages: list[dict],
    model: str,
) -> int:
    """Count prompt tokens for a message list by sending a 1-token request.

    Sends a non-streaming chat completion with ``max_tokens=1`` and reads
    ``usage.prompt_tokens`` from the response.

    Args:
        client: An open :class:`httpx.AsyncClient`.
        base_url: Base URL of the server.
        messages: The message list to send.
        model: Model ID to target.

    Returns:
        Number of prompt tokens, or ``0`` on error.
    """
    try:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 1,
                "stream": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return int((data.get("usage") or {}).get("prompt_tokens", 0))
    except Exception:
        return 0


async def stream_chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    messages: list[dict],
    model: str,
    max_tokens: int = 256,
    enable_thinking: Optional[bool] = None,
    extra_body: Optional[dict] = None,
) -> dict:
    """Send a streaming chat completion and collect per-token timing data.

    Tracks TTFT, per-token timestamps, accumulated content, finish reason,
    and usage (via ``stream_options: {"include_usage": True}``).

    Args:
        client: An open :class:`httpx.AsyncClient`.
        base_url: Base URL of the server.
        messages: The message list to send.
        model: Model ID to target.
        max_tokens: Maximum tokens to generate (default ``256``).
        enable_thinking: If not ``None``, passed as ``enable_thinking`` in the
            request body.
        extra_body: Optional extra keys merged into the request body.

    Returns:
        Dict with all :func:`compute_request_metrics` fields plus
        ``completion_tokens``, ``prompt_tokens``, ``finish_reason``,
        ``content``.
    """
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    if extra_body:
        body.update(extra_body)

    t_start = time.perf_counter()
    t_first_token: Optional[float] = None
    token_times: list[float] = []
    content_parts: list[str] = []
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None

    async with client.stream(
        "POST", f"{base_url}/v1/chat/completions", json=body
    ) as response:
        response.raise_for_status()
        async for raw_line in response.aiter_lines():
            parsed = parse_sse_line(raw_line)
            if parsed is None:
                continue
            if parsed.get("usage"):
                usage = parsed["usage"]
            if parsed.get("finish_reason"):
                finish_reason = parsed["finish_reason"]
            chunk_content = parsed.get("content", "")
            if chunk_content:
                now = time.perf_counter()
                if t_first_token is None:
                    t_first_token = now
                token_times.append(now)
                content_parts.append(chunk_content)

    t_end = time.perf_counter()
    if t_first_token is None:
        t_first_token = t_end

    prompt_tokens = int((usage or {}).get("prompt_tokens", 0))
    completion_tokens = int((usage or {}).get("completion_tokens", 0))

    metrics = compute_request_metrics(
        t_start=t_start,
        t_first_token=t_first_token,
        token_times=token_times,
        t_end=t_end,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    return {
        **metrics,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "finish_reason": finish_reason,
        "content": "".join(content_parts),
    }


# ---------------------------------------------------------------------------
# Task 5: Concurrent execution + validation + summary statistics
# ---------------------------------------------------------------------------


def validate_response(
    finish_reason: Optional[str],
    content: str,
    status_code: int,
) -> tuple[bool, str]:
    """Validate a single streaming response result.

    Args:
        finish_reason: The ``finish_reason`` from the final SSE chunk, or
            ``None`` if not received.
        content: The accumulated text content of the response.
        status_code: The HTTP status code of the response (use ``200`` for
            successful streaming requests).

    Returns:
        ``(is_valid, message)`` — ``is_valid`` is ``True`` when the response
        passes all checks; ``message`` is an empty string on success or a
        human-readable description of the first failure.
    """
    if status_code >= 400:
        return (False, f"HTTP error {status_code}")
    if finish_reason is None:
        return (False, "Missing finish_reason")
    if finish_reason == "length":
        return (False, "Truncated (finish_reason=length)")
    if not content:
        return (False, "Empty response content")
    return (True, "")


def validate_quality_checks(
    finish_reason: Optional[str],
    content: str,
    checks: Optional[dict],
    *,
    status_code: int = 200,
) -> tuple[bool, list[str]]:
    """Validate content against generic workload quality checks.

    Supported checks:
    - ``finish_reason``: string or list of allowed finish reasons
    - ``required_regex``: list of regex patterns that must match
    - ``forbidden_regex``: list of regex patterns that must not match
    - ``min_chars`` / ``max_chars``: length bounds
    - ``json``: when true, content must parse as JSON
    """
    basic_ok, basic_issue = validate_response(finish_reason, content, status_code)
    issues: list[str] = [] if basic_ok else [basic_issue]
    checks = checks or {}

    allowed_finish = checks.get("finish_reason")
    if allowed_finish is not None:
        allowed = (
            [allowed_finish]
            if isinstance(allowed_finish, str)
            else list(allowed_finish)
        )
        if finish_reason not in allowed:
            issues.append(
                f"finish_reason {finish_reason!r} not in allowed set {allowed!r}"
            )

    min_chars = checks.get("min_chars")
    if min_chars is not None and len(content) < int(min_chars):
        issues.append(f"content shorter than min_chars={min_chars}")

    max_chars = checks.get("max_chars")
    if max_chars is not None and len(content) > int(max_chars):
        issues.append(f"content longer than max_chars={max_chars}")

    for pattern in checks.get("required_regex", []) or []:
        try:
            if not re.search(str(pattern), content, re.MULTILINE):
                issues.append(f"required_regex did not match: {pattern}")
        except re.error as exc:
            issues.append(f"invalid required_regex {pattern!r}: {exc}")

    for pattern in checks.get("forbidden_regex", []) or []:
        try:
            if re.search(str(pattern), content, re.MULTILINE):
                issues.append(f"forbidden_regex matched: {pattern}")
        except re.error as exc:
            issues.append(f"invalid forbidden_regex {pattern!r}: {exc}")

    if checks.get("json"):
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            issues.append(f"content is not valid JSON: {exc}")

    return (not issues, issues)


def compute_summary_stats(values: list[float]) -> dict:
    """Compute summary statistics over a list of floats.

    Args:
        values: Non-empty list of floats to summarise.

    Returns:
        Dict with keys ``mean``, ``stddev``, ``min``, ``max``, ``p50``,
        ``p95``, ``p99``.  Percentiles use linear interpolation on sorted
        values.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        raise ValueError("Cannot compute summary stats on empty list")

    n = len(values)
    mean = statistics.mean(values)
    stddev = 0.0 if n == 1 else statistics.stdev(values)
    sorted_vals = sorted(values)

    def _percentile(p: float) -> float:
        if n == 1:
            return sorted_vals[0]
        # Linear interpolation: index = p/100 * (n-1)
        idx = p / 100.0 * (n - 1)
        lo = int(idx)
        hi = lo + 1
        if hi >= n:
            return sorted_vals[-1]
        frac = idx - lo
        return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])

    return {
        "mean": mean,
        "stddev": stddev,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "p50": _percentile(50),
        "p95": _percentile(95),
        "p99": _percentile(99),
    }


async def run_concurrent_requests(
    client: httpx.AsyncClient,
    base_url: str,
    prompts: list[list[dict]],
    model: str,
    concurrency: int,
    max_tokens: int = 256,
    enable_thinking: Optional[bool] = None,
    extra_body: Optional[dict] = None,
    do_validate: bool = True,
) -> list[dict]:
    """Fire ``concurrency`` concurrent streaming requests and collect results.

    Prompts are selected round-robin from ``prompts``.  All requests are
    launched simultaneously with :func:`asyncio.gather`.  Exceptions are
    caught per-task and wrapped in an error dict rather than propagated.

    Args:
        client: An open :class:`httpx.AsyncClient`.
        base_url: Base URL of the server.
        prompts: List of message dicts to cycle through.
        model: Model ID to target.
        concurrency: Number of simultaneous requests to fire.
        max_tokens: Maximum tokens to generate per request (default ``256``).
        enable_thinking: Passed through to :func:`stream_chat_completion`.
        extra_body: Passed through to :func:`stream_chat_completion`.
        do_validate: When ``True``, call :func:`validate_response` on each
            result and add a ``"validated"`` key.

    Returns:
        List of result dicts (one per request).  Each dict contains at minimum
        a ``"validated"`` key when ``do_validate`` is ``True``.
    """
    prompt_cycle = itertools.cycle(prompts)
    selected = [next(prompt_cycle) for _ in range(concurrency)]

    async def _single(messages: list[dict]) -> dict:
        try:
            result = await stream_chat_completion(
                client=client,
                base_url=base_url,
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                extra_body=extra_body,
            )
            if do_validate:
                is_valid, _ = validate_response(
                    finish_reason=result.get("finish_reason"),
                    content=result.get("content", ""),
                    status_code=200,
                )
                result["validated"] = is_valid
            return result
        except Exception as exc:
            err: dict = {
                "error": str(exc),
                "validated": False,
            }
            return err

    results = await asyncio.gather(*[_single(msg) for msg in selected])
    return list(results)


def _summary_or_empty(values: list[float]) -> dict:
    return compute_summary_stats(values) if values else {}


async def run_workload_case(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    workload: Workload,
    case: WorkloadCase,
    model: str,
    runtime: dict,
    hardware: dict,
    run_id: str,
    timestamp: str,
    scrape: bool = True,
    include_content: bool = False,
) -> dict:
    """Run one workload case and return a JSON-serializable result."""
    metrics_before = await scrape_metrics(client, base_url) if scrape else {}
    started_wall = datetime.now(timezone.utc).isoformat()

    try:
        result = await stream_chat_completion(
            client=client,
            base_url=base_url,
            messages=case.messages,
            model=model,
            max_tokens=int(case.max_tokens or workload.defaults.get("max_tokens", 256)),
            enable_thinking=case.enable_thinking,
            extra_body=case.extra_body,
        )
        error = ""
    except Exception as exc:
        result = {
            "ttft_ms": 0.0,
            "tpot_ms": 0.0,
            "e2e_latency_ms": 0.0,
            "gen_tps": 0.0,
            "prompt_tps": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "finish_reason": None,
            "content": "",
        }
        error = str(exc)

    metrics_after = await scrape_metrics(client, base_url) if scrape else {}
    status_after: dict = {}
    try:
        resp = await client.get(f"{base_url}/v1/status")
        resp.raise_for_status()
        status_after = resp.json()
    except Exception:
        status_after = {}

    cache_hits_delta = metrics_after.get("cache_hits", 0) - metrics_before.get(
        "cache_hits", 0
    )
    cache_misses_delta = metrics_after.get("cache_misses", 0) - metrics_before.get(
        "cache_misses", 0
    )
    tokens_saved_delta = metrics_after.get("tokens_saved", 0) - metrics_before.get(
        "tokens_saved", 0
    )

    content = str(result.get("content") or "")
    quality_ok, quality_issues = validate_quality_checks(
        result.get("finish_reason"),
        content,
        case.checks,
        status_code=500 if error else 200,
    )
    if error:
        quality_issues.append(f"request error: {error}")

    if case.policy_timeout_ms is None:
        within_policy_timeout = None
    elif error:
        within_policy_timeout = False
    else:
        within_policy_timeout = result["e2e_latency_ms"] <= case.policy_timeout_ms

    record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "started_at": started_wall,
        "workload": workload.name,
        "case_id": case.case_id,
        "tags": list(case.tags),
        "model_id": model,
        "runtime": runtime,
        "hardware": hardware,
        "request": {
            "max_tokens": int(
                case.max_tokens or workload.defaults.get("max_tokens", 256)
            ),
            "request_path": case.request_path,
            "enable_thinking": case.enable_thinking,
            "extra_body": case.extra_body or {},
            "message_count": len(case.messages),
        },
        "policy": {
            "timeout_ms": case.policy_timeout_ms,
            "within_timeout": within_policy_timeout,
            "note": "comparison-only unless your product contract explicitly requires it",
        },
        "metrics": {
            "ttft_ms": result["ttft_ms"],
            "tpot_ms": result["tpot_ms"],
            "e2e_latency_ms": result["e2e_latency_ms"],
            "gen_tps": result["gen_tps"],
            "prompt_tps": result["prompt_tps"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "cache_hits": cache_hits_delta,
            "cache_misses": cache_misses_delta,
            "tokens_saved": tokens_saved_delta,
            "metal": parse_status_response(status_after),
        },
        "quality": {
            "ok": quality_ok,
            "issues": quality_issues,
            "finish_reason": result.get("finish_reason"),
            "content_chars": len(content),
            "content_preview": content[:240],
        },
        "ok": quality_ok,
    }
    if include_content:
        record["quality"]["content"] = content
    return record


def summarize_workload_results(results: list[dict]) -> dict:
    """Aggregate workload case records into stable qualification summary stats."""
    latencies = [r["metrics"]["e2e_latency_ms"] for r in results]
    ttft = [r["metrics"]["ttft_ms"] for r in results]
    gen_tps = [r["metrics"]["gen_tps"] for r in results]
    quality_failures = [r for r in results if not r["quality"]["ok"]]
    policy_trials = [
        r for r in results if r["policy"].get("within_timeout") is not None
    ]
    policy_failures = [
        r for r in policy_trials if r["policy"].get("within_timeout") is False
    ]
    failures = [r for r in results if not r["quality"]["ok"]]
    return {
        "case_count": len(results),
        "passed": not failures,
        "failure_count": len(failures),
        "failure_rate": round(len(failures) / len(results), 4) if results else 0.0,
        "quality_passed": not quality_failures,
        "quality_failure_count": len(quality_failures),
        "policy_timeout_passed": not policy_failures if policy_trials else None,
        "policy_timeout_failure_count": (
            len(policy_failures) if policy_trials else None
        ),
        "latency_ms": _summary_or_empty(latencies),
        "ttft_ms": _summary_or_empty(ttft),
        "gen_tps": _summary_or_empty(gen_tps),
    }


async def run_bench_serve_workload(
    *,
    url: str,
    workload_path: str,
    model: Optional[str] = None,
    output_path: Optional[str] = None,
    output_format: str = "json",
    scrape: bool = True,
    include_content: bool = False,
    request_timeout_s: Optional[float] = 300.0,
) -> dict:
    """Run a declarative workload against a running server.

    This is the contract-style counterpart to prompt sweeps: it keeps product
    policy knobs in the manifest, records them as evidence, and measures what
    the server actually does before anyone promotes a model or feature stack.
    """
    workload = load_workload(workload_path)
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    timeout = httpx.Timeout(request_timeout_s) if request_timeout_s else None

    async with httpx.AsyncClient(timeout=timeout) as client:
        runtime = await auto_detect_runtime(client, url)
        hardware = detect_hardware_fingerprint()
        model_id = model or runtime.get("model_id", "")
        if not model_id:
            raise ValueError("could not determine model ID; pass --model")

        records = []
        for case in workload.cases:
            record = await run_workload_case(
                client,
                url,
                workload=workload,
                case=case,
                model=model_id,
                runtime=runtime,
                hardware=hardware,
                run_id=run_id,
                timestamp=timestamp,
                scrape=scrape,
                include_content=include_content,
            )
            records.append(record)

    payload = {
        "run_id": run_id,
        "timestamp": timestamp,
        "workload": {
            "name": workload.name,
            "description": workload.description,
            "path": str(Path(workload_path).expanduser()),
            "defaults": workload.defaults,
        },
        "transport": {
            "request_timeout_s": request_timeout_s,
            "note": "transport safety only; product policy timeouts live in workload cases",
        },
        "summary": summarize_workload_results(records),
        "results": records,
    }

    rendered = format_workload_payload(payload, output_format)
    if output_path:
        Path(output_path).expanduser().write_text(rendered)
        print(f"Workload results written to {output_path}")
    else:
        print(rendered)
    return payload


# ---------------------------------------------------------------------------
# Task 6: Output formatters
# ---------------------------------------------------------------------------

RESULT_COLUMNS: list[str] = [f.name for f in _dataclasses.fields(BenchServeResult)]

_TABLE_COLUMNS = [
    "prompt_set",
    "concurrency",
    "prompt_tokens",
    "ttft_ms",
    "tpot_ms",
    "gen_tps",
    "prompt_tps",
    "e2e_latency_ms",
    "validated",
]


def _result_to_dict(r: BenchServeResult) -> dict:
    """Convert a :class:`BenchServeResult` to an ordered dict.

    Returns an ``OrderedDict``-style plain ``dict`` whose keys follow the
    dataclass field declaration order (as listed in :data:`RESULT_COLUMNS`).
    """
    return {f.name: getattr(r, f.name) for f in _dataclasses.fields(r)}


def format_table(results: list[BenchServeResult]) -> str:
    """Render a human-readable terminal table of benchmark results.

    Only the columns in :data:`_TABLE_COLUMNS` are shown.  Float values are
    rounded to one decimal place.

    Args:
        results: List of :class:`BenchServeResult` instances.

    Returns:
        Formatted string using ``tabulate`` with ``tablefmt="simple"``.
    """
    rows = []
    for r in results:
        d = _result_to_dict(r)
        row = []
        for col in _TABLE_COLUMNS:
            val = d.get(col)
            if isinstance(val, float):
                val = round(val, 1)
            row.append(val)
        rows.append(row)
    return _tabulate(rows, headers=_TABLE_COLUMNS, tablefmt="simple")


def format_json(results: list[BenchServeResult]) -> str:
    """Serialize benchmark results as a JSON array.

    All fields from :data:`RESULT_COLUMNS` are included.

    Args:
        results: List of :class:`BenchServeResult` instances.

    Returns:
        JSON string with ``indent=2``.
    """
    return json.dumps([_result_to_dict(r) for r in results], indent=2)


def format_csv(results: list[BenchServeResult]) -> str:
    """Serialize benchmark results as CSV with a header row.

    All columns are included.

    Args:
        results: List of :class:`BenchServeResult` instances.

    Returns:
        CSV string (header + one row per result).
    """
    buf = io.StringIO()
    writer = csv_mod.DictWriter(buf, fieldnames=RESULT_COLUMNS)
    writer.writeheader()
    for r in results:
        writer.writerow(_result_to_dict(r))
    return buf.getvalue()


def _sql_escape(value) -> str:
    """Escape a Python value for use as a SQL literal.

    - ``None`` -> ``"NULL"``
    - ``bool`` -> ``"1"`` or ``"0"``
    - ``int`` / ``float`` -> string representation
    - ``str`` -> single-quoted with internal single-quotes doubled
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "NULL"
        return str(value)
    if isinstance(value, int):
        return str(value)
    # str
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


_SQL_SCHEMA = (
    "run_id TEXT, timestamp TEXT, tag TEXT, "
    "chip TEXT, gpu_cores INTEGER, memory_gb REAL, bandwidth_gbs REAL, os_version TEXT, "
    "model_id TEXT, model_type TEXT, engine_type TEXT, mtp_enabled BOOLEAN, "
    "specprefill BOOLEAN, kv_quant TEXT, cache_type TEXT, "
    "prompt_set TEXT, concurrency INTEGER, max_tokens INTEGER, enable_thinking BOOLEAN, "
    "extra_body TEXT, repetition INTEGER, prompt_tokens INTEGER, "
    "ttft_ms REAL, tpot_ms REAL, e2e_latency_ms REAL, "
    "gen_tps REAL, prompt_tps REAL, throughput_tps REAL, requests_per_s REAL, "
    "metal_active_gb REAL, metal_peak_gb REAL, metal_cache_gb REAL, "
    "cache_hits INTEGER, cache_misses INTEGER, cache_hit_rate REAL, tokens_saved INTEGER, "
    "validated BOOLEAN"
)


def format_sql(results: list[BenchServeResult]) -> str:
    """Emit a SQL ``CREATE TABLE IF NOT EXISTS`` statement and INSERT rows.

    The schema follows the exact column order defined in the bench-serve spec.

    Args:
        results: List of :class:`BenchServeResult` instances.

    Returns:
        SQL string containing the CREATE TABLE statement followed by one
        INSERT statement per result.
    """
    lines = [
        f"CREATE TABLE IF NOT EXISTS bench_serve ({_SQL_SCHEMA});",
    ]
    for r in results:
        d = _result_to_dict(r)
        values = ", ".join(_sql_escape(d[col]) for col in RESULT_COLUMNS)
        lines.append(f"INSERT INTO bench_serve VALUES ({values});")
    return "\n".join(lines)


WORKLOAD_RESULT_COLUMNS = [
    "run_id",
    "timestamp",
    "workload",
    "case_id",
    "tags",
    "model_id",
    "chip",
    "memory_gb",
    "os_version",
    "engine_type",
    "model_type",
    "mtp_enabled",
    "specprefill",
    "kv_quant",
    "cache_type",
    "request_max_tokens",
    "request_enable_thinking",
    "request_extra_body",
    "policy_timeout_ms",
    "within_policy_timeout",
    "ttft_ms",
    "tpot_ms",
    "e2e_latency_ms",
    "gen_tps",
    "prompt_tps",
    "prompt_tokens",
    "completion_tokens",
    "cache_hits",
    "cache_misses",
    "tokens_saved",
    "metal_active_gb",
    "metal_peak_gb",
    "metal_cache_gb",
    "quality_ok",
    "quality_issues",
    "finish_reason",
    "content_chars",
    "content_preview",
]

_WORKLOAD_TABLE_COLUMNS = [
    "case_id",
    "tags",
    "quality_ok",
    "within_policy_timeout",
    "ttft_ms",
    "gen_tps",
    "e2e_latency_ms",
    "cache_hits",
    "tokens_saved",
    "finish_reason",
]


def _workload_record_to_row(record: dict) -> dict:
    runtime = record.get("runtime") or {}
    hardware = record.get("hardware") or {}
    request = record.get("request") or {}
    policy = record.get("policy") or {}
    metrics = record.get("metrics") or {}
    metal = metrics.get("metal") or {}
    quality = record.get("quality") or {}
    return {
        "run_id": record.get("run_id", ""),
        "timestamp": record.get("timestamp", ""),
        "workload": record.get("workload", ""),
        "case_id": record.get("case_id", ""),
        "tags": ",".join(record.get("tags") or []),
        "model_id": record.get("model_id", ""),
        "chip": hardware.get("chip", ""),
        "memory_gb": hardware.get("memory_gb", 0.0),
        "os_version": hardware.get("os_version", ""),
        "engine_type": runtime.get("engine_type", ""),
        "model_type": runtime.get("model_type", ""),
        "mtp_enabled": runtime.get("mtp_enabled", False),
        "specprefill": runtime.get("specprefill", False),
        "kv_quant": runtime.get("kv_quant", ""),
        "cache_type": runtime.get("cache_type", ""),
        "request_max_tokens": request.get("max_tokens"),
        "request_enable_thinking": request.get("enable_thinking"),
        "request_extra_body": json.dumps(
            request.get("extra_body") or {}, sort_keys=True
        ),
        "policy_timeout_ms": policy.get("timeout_ms"),
        "within_policy_timeout": policy.get("within_timeout"),
        "ttft_ms": metrics.get("ttft_ms", 0.0),
        "tpot_ms": metrics.get("tpot_ms", 0.0),
        "e2e_latency_ms": metrics.get("e2e_latency_ms", 0.0),
        "gen_tps": metrics.get("gen_tps", 0.0),
        "prompt_tps": metrics.get("prompt_tps", 0.0),
        "prompt_tokens": metrics.get("prompt_tokens", 0),
        "completion_tokens": metrics.get("completion_tokens", 0),
        "cache_hits": metrics.get("cache_hits", 0),
        "cache_misses": metrics.get("cache_misses", 0),
        "tokens_saved": metrics.get("tokens_saved", 0),
        "metal_active_gb": metal.get("metal_active_gb", 0.0),
        "metal_peak_gb": metal.get("metal_peak_gb", 0.0),
        "metal_cache_gb": metal.get("metal_cache_gb", 0.0),
        "quality_ok": quality.get("ok", False),
        "quality_issues": json.dumps(quality.get("issues") or []),
        "finish_reason": quality.get("finish_reason"),
        "content_chars": quality.get("content_chars", 0),
        "content_preview": quality.get("content_preview", ""),
    }


def format_workload_table(payload: dict) -> str:
    rows = []
    for record in payload.get("results") or []:
        row = _workload_record_to_row(record)
        rows.append(
            [
                round(value, 1) if isinstance(value, float) else value
                for value in (row[col] for col in _WORKLOAD_TABLE_COLUMNS)
            ]
        )
    return _tabulate(rows, headers=_WORKLOAD_TABLE_COLUMNS, tablefmt="simple")


def format_workload_json(payload: dict) -> str:
    return json.dumps(payload, indent=2)


def format_workload_csv(payload: dict) -> str:
    buf = io.StringIO()
    writer = csv_mod.DictWriter(buf, fieldnames=WORKLOAD_RESULT_COLUMNS)
    writer.writeheader()
    for record in payload.get("results") or []:
        writer.writerow(_workload_record_to_row(record))
    return buf.getvalue()


_WORKLOAD_SQL_SCHEMA = (
    "run_id TEXT, timestamp TEXT, workload TEXT, case_id TEXT, tags TEXT, "
    "model_id TEXT, chip TEXT, memory_gb REAL, os_version TEXT, "
    "engine_type TEXT, model_type TEXT, mtp_enabled BOOLEAN, specprefill BOOLEAN, "
    "kv_quant TEXT, cache_type TEXT, request_max_tokens INTEGER, "
    "request_enable_thinking BOOLEAN, request_extra_body TEXT, "
    "policy_timeout_ms INTEGER, within_policy_timeout BOOLEAN, "
    "ttft_ms REAL, tpot_ms REAL, e2e_latency_ms REAL, gen_tps REAL, "
    "prompt_tps REAL, prompt_tokens INTEGER, completion_tokens INTEGER, "
    "cache_hits INTEGER, cache_misses INTEGER, tokens_saved INTEGER, "
    "metal_active_gb REAL, metal_peak_gb REAL, metal_cache_gb REAL, "
    "quality_ok BOOLEAN, quality_issues TEXT, finish_reason TEXT, "
    "content_chars INTEGER, content_preview TEXT"
)


def format_workload_sql(payload: dict) -> str:
    lines = [
        f"CREATE TABLE IF NOT EXISTS bench_serve_workload ({_WORKLOAD_SQL_SCHEMA});",
    ]
    for record in payload.get("results") or []:
        row = _workload_record_to_row(record)
        values = ", ".join(_sql_escape(row[col]) for col in WORKLOAD_RESULT_COLUMNS)
        lines.append(f"INSERT INTO bench_serve_workload VALUES ({values});")
    return "\n".join(lines)


def format_workload_payload(payload: dict, fmt: str = "json") -> str:
    if fmt == "json":
        return format_workload_json(payload)
    if fmt == "csv":
        return format_workload_csv(payload)
    if fmt == "sql":
        return format_workload_sql(payload)
    if fmt == "table":
        return format_workload_table(payload)
    raise ValueError(f"Unsupported workload output format: {fmt}")


# ---------------------------------------------------------------------------
# Task 7: Main async orchestrator
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


async def run_bench_serve(
    url: str = "http://127.0.0.1:8080",
    model: Optional[str] = None,
    prompt_sets: list[str] = None,
    prompt_file: Optional[str] = None,
    concurrencies: list[int] = None,
    max_tokens: int = 256,
    repetitions: int = 3,
    warmup: int = 1,
    thinking_values: list[Optional[bool]] = None,
    extra_bodies: list[str] = None,
    output_path: Optional[str] = None,
    fmt: str = "table",
    do_validate: bool = True,
    scrape: bool = True,
    tag: Optional[str] = None,
    override_fields: Optional[dict] = None,
    system_prompt_file: Optional[str] = None,
    skip_preflight_token_count: bool = False,
) -> list[BenchServeResult]:
    """Run the full bench-serve sweep against a running vllm-mlx server.

    Args:
        url: Base URL of the server.
        model: Model ID to use. If ``None``, auto-detected from the server.
        prompt_sets: List of prompt set names or paths. Defaults to
            ``["short", "medium", "long"]``.
        prompt_file: Optional path to an extra prompt file to include.
        concurrencies: Concurrency levels to sweep. Defaults to ``[1, 4]``.
        max_tokens: Maximum tokens to generate per request.
        repetitions: Number of repetitions per sweep config.
        warmup: Number of warmup rounds before the first measured repetition.
        thinking_values: Values for ``enable_thinking``. Defaults to
            ``[None]``.
        extra_bodies: JSON strings for extra body parameters. Defaults to
            ``[""]`` (no extra body).
        output_path: File path to write results to. If ``None``, prints to
            stdout.
        fmt: Output format — one of ``"table"``, ``"json"``, ``"csv"``,
            ``"sql"``.
        do_validate: Whether to validate each response.
        scrape: Whether to scrape ``/metrics`` before and after each run.
        tag: Optional tag string stored in every result row.
        override_fields: Dict of field names to override on every result.

    Returns:
        List of :class:`BenchServeResult` instances.
    """
    # 1. Set defaults
    if prompt_sets is None:
        prompt_sets = ["short", "medium", "long"]
    if concurrencies is None:
        concurrencies = [1, 4]
    if thinking_values is None:
        thinking_values = [None]
    if extra_bodies is None:
        extra_bodies = [""]
    if override_fields is None:
        override_fields = {}

    # 2. Generate run_id and timestamp
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    # 3. Open HTTP client
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        # 4. Auto-detect runtime and hardware
        print(f"Connecting to {url}...")
        runtime = await auto_detect_runtime(client, url)
        hw = detect_hardware_fingerprint()

        # 5. Resolve model_id
        model_id = model or runtime.get("model_id", "")
        if not model_id:
            print(
                "Error: could not determine model ID. Use --model to specify.",
                file=sys.stderr,
            )
            return []

        # 6. Print hardware and runtime info
        print(
            f"Hardware: {hw.get('chip', 'unknown')} / {hw.get('memory_gb', 0):.0f}GB / {hw.get('os_version', '')}"
        )
        print(
            f"Runtime:  model={model_id}  engine={runtime.get('engine_type', '')}  cache={runtime.get('cache_type', '')}"
        )

        # 7. Load prompts
        all_prompts: dict[str, list[list[dict]]] = {}
        for ps in prompt_sets:
            try:
                all_prompts[ps] = load_prompt_set(ps)
            except FileNotFoundError as exc:
                print(f"Warning: skipping prompt set '{ps}': {exc}", file=sys.stderr)
        if prompt_file:
            try:
                all_prompts[prompt_file] = load_prompt_set(prompt_file)
            except FileNotFoundError as exc:
                print(
                    f"Warning: skipping prompt file '{prompt_file}': {exc}",
                    file=sys.stderr,
                )

        if not all_prompts:
            print("Error: no prompt sets could be loaded.", file=sys.stderr)
            return []

        # 7b. If --system-prompt-file given, prepend that system message to
        # every prompt across every set. This is the warm-prompts path: the
        # server was started with the same system in its warm-up file, so
        # every request here hits the prefix cache.
        if system_prompt_file:
            sys_path = Path(system_prompt_file).expanduser()
            if not sys_path.exists():
                print(
                    f"Error: --system-prompt-file not found: {sys_path}",
                    file=sys.stderr,
                )
                return []
            sys_content = sys_path.read_text()
            system_msg = {"role": "system", "content": sys_content}
            for ps, prompts in all_prompts.items():
                patched: list[list[dict]] = []
                for msgs in prompts:
                    # Do not double-prepend if the prompt already has a system
                    # as its first message.
                    if msgs and msgs[0].get("role") == "system":
                        patched.append(msgs)
                    else:
                        patched.append([system_msg] + msgs)
                all_prompts[ps] = patched
            print(f"System prompt prepended from {sys_path} ({len(sys_content)} chars)")

        # 8. Token-count first prompt from each set.
        # This sends a non-streaming max_tokens=1 request per prompt set, which
        # populates the server's prefix cache with the full prompt. That is
        # harmless for ordinary benchmarking but DEFEATS cold-vs-warm
        # comparisons (both paths end up warm after the pre-flight). Skip it
        # when --skip-preflight-token-count is set; prompt_tokens will be
        # populated from the first measured request's usage instead.
        prompt_token_counts: dict[str, int] = {}
        if skip_preflight_token_count:
            for ps in all_prompts:
                prompt_token_counts[ps] = 0
        else:
            for ps, prompts in all_prompts.items():
                try:
                    count = await count_prompt_tokens(client, url, prompts[0], model_id)
                    prompt_token_counts[ps] = count
                except Exception:
                    prompt_token_counts[ps] = 0

        # 9. Expand sweep
        sweep = expand_sweep(
            list(all_prompts.keys()),
            concurrencies,
            thinking_values,
            extra_bodies,
            repetitions,
        )

        # Account for warmup rounds: insert warmup configs at rep==0 boundaries.
        # We handle warmup inline during the sweep by tracking which
        # (ps, conc, think, eb) combos have been warmed up.
        total_runs = len(sweep)
        warmup_note = f" (+ {warmup} warmup per config)" if warmup > 0 else ""
        print(f"Total runs: {total_runs}{warmup_note}")

        results: list[BenchServeResult] = []
        warmed_up: set[tuple] = set()

        # 11. Iterate over sweep
        for ps, conc, think, eb, rep in sweep:
            prompts = all_prompts[ps]

            # Parse extra_body
            extra_body_dict: Optional[dict] = None
            if eb:
                try:
                    extra_body_dict = json.loads(eb)
                except json.JSONDecodeError:
                    extra_body_dict = None

            # Format progress label
            think_label = f"think={think}" if think is not None else ""
            eb_label = f"eb={eb[:20]}" if eb else ""
            label_parts = [ps, f"conc={conc}", f"rep={rep}"]
            if think_label:
                label_parts.append(think_label)
            if eb_label:
                label_parts.append(eb_label)
            label = " ".join(label_parts)

            # d. Warmup on first rep of each unique config
            warmup_key = (ps, conc, think, eb)
            if rep == 0 and warmup_key not in warmed_up and warmup > 0:
                warmed_up.add(warmup_key)
                for _ in range(warmup):
                    try:
                        await run_concurrent_requests(
                            client=client,
                            base_url=url,
                            prompts=prompts,
                            model=model_id,
                            concurrency=conc,
                            max_tokens=max_tokens,
                            enable_thinking=think,
                            extra_body=extra_body_dict,
                            do_validate=False,
                        )
                    except Exception:
                        pass

            # e. Scrape /metrics before
            metrics_before: dict = {}
            if scrape:
                metrics_before = await scrape_metrics(client, url)

            # f. Run concurrent requests
            req_results = await run_concurrent_requests(
                client=client,
                base_url=url,
                prompts=prompts,
                model=model_id,
                concurrency=conc,
                max_tokens=max_tokens,
                enable_thinking=think,
                extra_body=extra_body_dict,
                do_validate=do_validate,
            )

            # g. Scrape /metrics after, compute cache delta
            metrics_after: dict = {}
            if scrape:
                metrics_after = await scrape_metrics(client, url)

            cache_hits_delta = metrics_after.get("cache_hits", 0) - metrics_before.get(
                "cache_hits", 0
            )
            cache_misses_delta = metrics_after.get(
                "cache_misses", 0
            ) - metrics_before.get("cache_misses", 0)
            tokens_saved_delta = metrics_after.get(
                "tokens_saved", 0
            ) - metrics_before.get("tokens_saved", 0)
            total_events = cache_hits_delta + cache_misses_delta
            cache_hit_rate = (
                cache_hits_delta / total_events if total_events > 0 else 0.0
            )

            # h. Get metal memory from /v1/status
            metal_active_gb = runtime.get("metal_active_gb", 0.0)
            metal_peak_gb = runtime.get("metal_peak_gb", 0.0)
            metal_cache_gb = runtime.get("metal_cache_gb", 0.0)
            try:
                resp = await client.get(f"{url}/v1/status")
                resp.raise_for_status()
                status_data = parse_status_response(resp.json())
                metal_active_gb = status_data.get("metal_active_gb", metal_active_gb)
                metal_peak_gb = status_data.get("metal_peak_gb", metal_peak_gb)
                metal_cache_gb = status_data.get("metal_cache_gb", metal_cache_gb)
            except Exception:
                pass

            # i. Aggregate per-request metrics
            valid_results = [r for r in req_results if "error" not in r]
            if not valid_results:
                # All requests errored — build a failed result
                result_obj = BenchServeResult(
                    run_id=run_id,
                    timestamp=timestamp,
                    tag=tag or "",
                    # Hardware
                    chip=hw.get("chip", ""),
                    gpu_cores=hw.get("gpu_cores", 0),
                    memory_gb=hw.get("memory_gb", 0.0),
                    bandwidth_gbs=hw.get("bandwidth_gbs", 0.0),
                    os_version=hw.get("os_version", ""),
                    # Runtime
                    model_id=model_id,
                    model_type=runtime.get("model_type", ""),
                    engine_type=runtime.get("engine_type", ""),
                    mtp_enabled=runtime.get("mtp_enabled", False),
                    specprefill=runtime.get("specprefill", False),
                    kv_quant=runtime.get("kv_quant", ""),
                    cache_type=runtime.get("cache_type", ""),
                    # Config
                    prompt_set=ps,
                    concurrency=conc,
                    max_tokens=max_tokens,
                    enable_thinking=think,
                    extra_body=eb,
                    repetition=rep,
                    prompt_tokens=prompt_token_counts.get(ps, 0),
                    # Latency / throughput all zero
                    validated=False,
                )
                print(f"  {label}: FAIL (all requests errored)")
            else:

                def _mean(key: str) -> float:
                    vals = [
                        r[key] for r in valid_results if key in r and r[key] is not None
                    ]
                    return statistics.mean(vals) if vals else 0.0

                mean_ttft = _mean("ttft_ms")
                mean_tpot = _mean("tpot_ms")
                mean_gen_tps = _mean("gen_tps")
                mean_prompt_tps = _mean("prompt_tps")
                mean_e2e = _mean("e2e_latency_ms")

                total_completion_tokens = sum(
                    r.get("completion_tokens", 0) for r in valid_results
                )
                max_e2e_seconds = (
                    max(
                        (r.get("e2e_latency_ms", 0.0) for r in valid_results),
                        default=0.0,
                    )
                    / 1000.0
                )
                throughput_tps = (
                    total_completion_tokens / max_e2e_seconds
                    if max_e2e_seconds > 0
                    else 0.0
                )
                requests_per_s = conc / max_e2e_seconds if max_e2e_seconds > 0 else 0.0

                all_validated = all(r.get("validated", True) for r in valid_results)

                result_obj = BenchServeResult(
                    run_id=run_id,
                    timestamp=timestamp,
                    tag=tag or "",
                    # Hardware
                    chip=hw.get("chip", ""),
                    gpu_cores=hw.get("gpu_cores", 0),
                    memory_gb=hw.get("memory_gb", 0.0),
                    bandwidth_gbs=hw.get("bandwidth_gbs", 0.0),
                    os_version=hw.get("os_version", ""),
                    # Runtime
                    model_id=model_id,
                    model_type=runtime.get("model_type", ""),
                    engine_type=runtime.get("engine_type", ""),
                    mtp_enabled=runtime.get("mtp_enabled", False),
                    specprefill=runtime.get("specprefill", False),
                    kv_quant=runtime.get("kv_quant", ""),
                    cache_type=runtime.get("cache_type", ""),
                    # Config
                    prompt_set=ps,
                    concurrency=conc,
                    max_tokens=max_tokens,
                    enable_thinking=think,
                    extra_body=eb,
                    repetition=rep,
                    prompt_tokens=prompt_token_counts.get(ps, 0),
                    # Latency
                    ttft_ms=mean_ttft,
                    tpot_ms=mean_tpot,
                    e2e_latency_ms=mean_e2e,
                    # Throughput
                    gen_tps=mean_gen_tps,
                    prompt_tps=mean_prompt_tps,
                    throughput_tps=throughput_tps,
                    requests_per_s=requests_per_s,
                    # Memory
                    metal_active_gb=metal_active_gb,
                    metal_peak_gb=metal_peak_gb,
                    metal_cache_gb=metal_cache_gb,
                    # Cache
                    cache_hits=cache_hits_delta,
                    cache_misses=cache_misses_delta,
                    cache_hit_rate=cache_hit_rate,
                    tokens_saved=tokens_saved_delta,
                    # Validation
                    validated=all_validated,
                )

                status = "PASS" if all_validated else "FAIL"
                print(
                    f"  {label}: TTFT={mean_ttft:.0f}ms  TPS={mean_gen_tps:.1f}  {status}"
                )

            # j. Apply override_fields
            for field_name, field_val in override_fields.items():
                if hasattr(result_obj, field_name):
                    setattr(result_obj, field_name, field_val)

            results.append(result_obj)

        # 12. Format output
        formatters = {
            "table": format_table,
            "json": format_json,
            "csv": format_csv,
            "sql": format_sql,
        }
        formatter = formatters.get(fmt, format_table)
        output = formatter(results)

        # 13. Write to file or stdout
        if output_path:
            Path(output_path).write_text(output)
            print(f"\nResults written to {output_path}")
        else:
            print()
            print(output)

        return results
