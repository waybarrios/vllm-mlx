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

import itertools
import json
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

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
