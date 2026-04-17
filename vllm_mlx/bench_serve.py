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
import io
import itertools
import json
import platform
import re
import statistics
import time
from dataclasses import dataclass, field
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

    gen_duration = t_end - t_first_token
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

import dataclasses as _dataclasses  # noqa: E402 — local alias avoids shadowing

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
    if isinstance(value, (int, float)):
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
