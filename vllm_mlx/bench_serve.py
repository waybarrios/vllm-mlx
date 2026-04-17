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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from tabulate import tabulate as _tabulate

# ---------------------------------------------------------------------------
# Prompt set loading
# ---------------------------------------------------------------------------

_BUILTIN_DIR = Path(__file__).parent / "bench_serve_prompts"
_BUILTIN_NAMES = {"short", "medium", "long", "thinking"}


def load_prompt_set(name_or_path: str) -> list[dict]:
    """Load a prompt set by builtin name or file path.

    Builtin sets (``short``, ``medium``, ``long``, ``thinking``) are loaded
    from the ``bench_serve_prompts/`` directory next to this module.  Any
    other value is treated as a filesystem path and loaded directly.

    Args:
        name_or_path: One of the builtin set names, or an absolute / relative
            path to a JSON file containing a list of message dicts.

    Returns:
        A list of message dicts (each with at minimum a ``"role"`` key).

    Raises:
        FileNotFoundError: If ``name_or_path`` is not a known builtin name and
            the path does not exist, or if a builtin name is requested but its
            JSON file is missing from the package.
    """
    if name_or_path in _BUILTIN_NAMES:
        target = _BUILTIN_DIR / f"{name_or_path}.json"
        if not target.exists():
            raise FileNotFoundError(
                f"Builtin prompt set '{name_or_path}' not found at {target}"
            )
        with target.open() as fh:
            return json.load(fh)

    # Treat as a custom filesystem path.
    path = Path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Unknown prompt set name or missing file: '{name_or_path}'. "
            f"Builtin names are: {sorted(_BUILTIN_NAMES)}"
        )
    with path.open() as fh:
        return json.load(fh)


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
    prompts: list[dict],
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

    async def _single(messages: dict) -> dict:
        try:
            result = await stream_chat_completion(
                client=client,
                base_url=base_url,
                messages=[messages],
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
        all_prompts: dict[str, list[dict]] = {}
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

        # 8. Token-count first prompt from each set
        prompt_token_counts: dict[str, int] = {}
        for ps, prompts in all_prompts.items():
            try:
                count = await count_prompt_tokens(client, url, [prompts[0]], model_id)
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
        print(f"Total runs: {total_runs} (+ warmup on first rep of each config)")

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
