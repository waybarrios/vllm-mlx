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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
