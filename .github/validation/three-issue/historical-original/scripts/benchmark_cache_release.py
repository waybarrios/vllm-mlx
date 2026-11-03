#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apple-only, fresh-process explicit cache DELETE memory attribution harness.

Run identically on base and candidate, e.g. from an editable project environment:
  python scripts/benchmark_cache_release.py --route prefix > prefix.jsonl
  python scripts/benchmark_cache_release.py --route all > all.jsonl
  python scripts/benchmark_cache_release.py --model /existing/snapshot --route prefix

Default mode is SYNTHETIC KV in a real PrefixCacheManager, not production-model
validation. --model uses a real BatchedEngine and generation from a LOCAL snapshot
only, kept loaded through all measurements. Offline mode prevents downloads.
Repeat each command three times in separate processes for median comparisons.
No post-DELETE test-side clear/GC/synchronization can rescue the baseline.
Thresholds are attribution gates, not promises about macOS accounting latency.
"""

import argparse
import asyncio
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
import time

MIB = 1024**2


class _RusageInfoV0(ctypes.Structure):
    # RUSAGE_INFO_V0, current (not lifetime-peak) resident/physical bytes.
    # https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/resource.h
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
        )
    ]


def _os_memory():
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    call = libproc.proc_pid_rusage
    call.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    call.restype = ctypes.c_int
    info = _RusageInfoV0()
    if call(os.getpid(), 0, ctypes.byref(info)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    return {"rss_bytes": info.resident_size, "footprint_bytes": info.phys_footprint}


def _emit(event, **data):
    print(
        json.dumps({"event": event, "monotonic_s": time.monotonic(), **data}),
        flush=True,
    )


def _snapshot(mx, engine, phase, cycle=0):
    active, cached = mx.get_active_memory(), mx.get_cache_memory()
    sample = {
        "phase": phase,
        "cycle": cycle,
        "active_bytes": active,
        "cache_bytes": cached,
        "active_plus_cache_bytes": active + cached,
        **_os_memory(),
    }
    if engine is not None:
        sample["engine"] = engine.get_stats()
        sample["prefix_cache"] = engine.get_cache_stats()
    _emit("sample", **sample)
    return sample


def _entry_count(stats):
    """Read the existing cache stats schemas; unknown is not an empty cache."""
    if not isinstance(stats, dict):
        return None
    for key in ("entry_count", "entries", "size"):
        if isinstance(stats.get(key), int):
            return stats[key]
    return _entry_count(stats.get("prefix_cache"))


class _SyntheticEngine:
    """Single event-loop owner; NOT a production-engine/worker substitute."""

    def __init__(self, mx):
        from vllm_mlx.prefix_cache import PrefixCacheManager

        self.model = mx.arange(4 * MIB, dtype=mx.float32)  # 16MiB persistent owner
        mx.eval(self.model)
        self.cache = PrefixCacheManager(self.model)

    def fill(self, mx, size):
        from mlx_lm.models.cache import KVCache

        layer = KVCache()
        shape = (1, 1, size // 8 // 128, 128)
        layer.keys = mx.arange(size // 8, dtype=mx.float32).reshape(shape)
        layer.values = (mx.arange(size // 8, dtype=mx.float32) + 1).reshape(shape)
        mx.eval(layer.keys, layer.values)
        layer.offset = shape[-2]
        self.cache.store_cache([1, 2, 3], [layer])
        # No layer/array aliases escape; cache is now the sole KV owner.

    def clear_prefix_cache(self):
        self.cache.clear()

    async def _clear_prefix_cache_on_owner(self):
        self.clear_prefix_cache()

    def clear_runtime_caches(self):
        self.clear_prefix_cache()
        return {"prefix_cache": True}

    async def _clear_runtime_caches_on_owner(self):
        return self.clear_runtime_caches()

    def get_stats(self):
        return {"num_running": 0, "num_waiting": 0, "steps_executed": 0}

    def get_cache_stats(self):
        return {"prefix_cache": {"entry_count": len(self.cache._lru)}}


async def _quiesce(mx, engine):
    """Benchmark only: finish cleanup and synchronize actual producing streams."""
    deadline = time.monotonic() + 30
    while True:
        stats = engine.get_stats()
        if not stats.get("num_running", 0) and not stats.get("num_waiting", 0):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("engine did not become idle")
        await asyncio.sleep(0.01)

    def synchronize():
        streams = [mx.default_stream(mx.default_device())]
        if not isinstance(engine, _SyntheticEngine):
            scheduler = (
                engine._mllm_scheduler
                if engine.is_mllm
                else engine._engine.engine.scheduler
            )
            batch = scheduler.batch_generator
            if batch is not None and getattr(batch, "_stream", None) is not None:
                streams.append(batch._stream)
            for name in ("mlx_lm.generate", "mlx_vlm.generate"):
                module = sys.modules.get(name)
                if module is not None and hasattr(module, "generation_stream"):
                    streams.append(module.generation_stream)
        # Handles may coincide; repeated synchronization is harmless, not a flush.
        for stream in streams:
            mx.synchronize(stream)

    if isinstance(engine, _SyntheticEngine) or engine.is_mllm:
        synchronize()
    else:
        await engine._run_on_generation_worker(synchronize)


async def _generate(engine, text):
    output = None
    async for output in engine.stream_chat(
        [{"role": "user", "content": text}], max_tokens=16, temperature=0
    ):
        pass
    if output is None or not output.finished:
        raise RuntimeError("generation did not finish")
    if output.finish_reason == "error":
        raise RuntimeError("generation failed")
    _emit(
        "generation",
        prompt_tokens=output.prompt_tokens,
        completion_tokens=output.completion_tokens,
        output_sha256=hashlib.sha256(output.text.encode()).hexdigest(),
    )


async def _run(args):
    import httpx
    import mlx.core as mx

    import vllm_mlx.server as server

    if not mx.metal.is_available():
        raise RuntimeError("BLOCKED: Metal backend unavailable")
    mx.set_default_device(mx.gpu)
    versions = {}
    for name in ("mlx", "mlx-lm", "mlx-vlm", "vllm-mlx", "httpx"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    _emit(
        "config",
        platform=platform.platform(),
        python=sys.version,
        versions=versions,
        device=getattr(mx, "device_info", mx.metal.device_info)(),
        mode="production-model" if args.model else "synthetic-kv-only",
        arguments=vars(args),
        command=sys.argv,
        stream_policy="synchronize default and actual generator streams on owner before DELETE",
    )
    _snapshot(mx, None, "before_load")
    engine = None
    try:
        if args.model:
            from vllm_mlx.engine.batched import BatchedEngine
            from vllm_mlx.scheduler import SchedulerConfig

            engine = BatchedEngine(
                str(Path(args.model).resolve()),
                trust_remote_code=False,
                scheduler_config=SchedulerConfig(cache_memory_mb=args.cache_limit_mib),
            )
            await engine.start()
        else:
            engine = _SyntheticEngine(mx)
        server._engine = engine
        server._warm_prompts_path = None
        server._api_key = None
        await _quiesce(mx, engine)
        _snapshot(mx, engine, "after_load")
        model = engine._model if args.model else engine.model
        model_id = id(model)
        # One setup-only drain before any fill. Never repeated between cycles.
        old_cap = mx.set_cache_limit(args.cache_limit_mib * MIB)
        _emit("allocator_policy", previous_cap_bytes=old_cap)
        mx.clear_cache()
        resident = _snapshot(mx, engine, "resident_baseline")
        failures = []
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://benchmark"
        ) as client:
            route = "/v1/cache/prefix" if args.route == "prefix" else "/v1/cache"
            for cycle in range(1, args.cycles + 1):
                if args.model:
                    for request in range(args.requests):
                        text = (
                            f"Cycle {cycle} request {request}. "
                            + "memory cache test " * args.prompt_repeats
                        )
                        await _generate(engine, text)
                else:
                    engine.fill(mx, args.kv_mib * MIB)
                await _quiesce(mx, engine)
                filled = _snapshot(mx, engine, "filled_idle", cycle)
                size = (
                    filled["active_bytes"] - resident["active_bytes"]
                    if args.model
                    else args.kv_mib * MIB
                )
                response = await client.delete(route)
                response.raise_for_status()
                if response.json().get("status") != "cleared":
                    raise RuntimeError(f"DELETE did not clear caches: {response.text}")
                _emit("delete", cycle=cycle, response=response.json())
                start = time.monotonic()
                # No GPU work, GC, allocator clear or synchronization in this window.
                samples = [_snapshot(mx, engine, "after_clear_0s", cycle)]
                for delay in (1, 5, 10, 60):
                    if delay <= args.settle_seconds:
                        await asyncio.sleep(max(0, start + delay - time.monotonic()))
                        samples.append(
                            _snapshot(mx, engine, f"after_clear_{delay}s", cycle)
                        )
                settled = samples[-1]
                checks = {
                    "sufficient_workload": size >= 16 * MIB,
                    "cache_was_populated": (_entry_count(filled["prefix_cache"]) or 0)
                    > 0,
                    "cache_entries_cleared": _entry_count(settled["prefix_cache"]) == 0,
                    "active_drop": filled["active_bytes"] - settled["active_bytes"]
                    >= 0.8 * size,
                    "pool_release": filled["active_plus_cache_bytes"]
                    - settled["active_plus_cache_bytes"]
                    >= 0.8 * size,
                    "unused_pool_empty": samples[0]["cache_bytes"] == 0,
                    "rss_drop": filled["rss_bytes"] - settled["rss_bytes"]
                    >= 0.5 * size,
                    "footprint_drop": filled["footprint_bytes"]
                    - settled["footprint_bytes"]
                    >= 0.5 * size,
                    "rss_plateau": settled["rss_bytes"]
                    <= resident["rss_bytes"] + 32 * MIB,
                    "footprint_plateau": settled["footprint_bytes"]
                    <= resident["footprint_bytes"] + 32 * MIB,
                    "idle_steps_unchanged": filled["engine"].get("steps_executed")
                    is not None
                    and settled["engine"].get("steps_executed")
                    == filled["engine"].get("steps_executed"),
                    "model_resident": model_id
                    == id(engine._model if args.model else engine.model),
                }
                _emit("cycle_checks", cycle=cycle, attributed_bytes=size, checks=checks)
                failures.extend(
                    f"cycle {cycle}: {key}" for key, ok in checks.items() if not ok
                )
                # Idempotence is tested only AFTER the last accounting sample.
                again = await client.delete(route)
                again.raise_for_status()
            if args.model:
                await _generate(engine, "Reply with OK.")
            else:
                if float(engine.model[1].item()) != 1.0:
                    failures.append("persistent synthetic owner corrupted")
            _emit(
                "summary", outcome="failed" if failures else "passed", failures=failures
            )
            return 1 if failures else 0
    finally:
        # Shutdown follows all measured samples; unload cannot explain a drop.
        if args.model and engine is not None:
            await engine.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", help="Already-present local model directory; never a Hub ID"
    )
    parser.add_argument("--route", choices=("prefix", "all"), default="prefix")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--kv-mib", type=int, default=256)
    parser.add_argument("--cache-limit-mib", type=int, default=1024)
    parser.add_argument("--settle-seconds", type=int, choices=(10, 60), default=60)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--prompt-repeats", type=int, default=1024)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        _emit(
            "blocked",
            reason="requires actual Apple Silicon macOS/Metal",
            platform=platform.platform(),
        )
        return 2
    if args.cycles < 3 or args.kv_mib < 16 or args.kv_mib % 16:
        parser.error("require >=3 cycles and KV MiB a positive multiple of 16")
    if args.cache_limit_mib < max(1024, 2 * args.kv_mib):
        parser.error("allocator cap must be >=1GiB and >=2x synthetic KV size")
    if args.requests < 1 or args.prompt_repeats < 1:
        parser.error("requests and prompt-repeats must be positive")
    if args.model and not (
        Path(args.model).is_dir() and (Path(args.model) / "config.json").is_file()
    ):
        parser.error(
            "--model must name an existing local snapshot containing config.json"
        )
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    # Ensure a direct source-checkout invocation imports this checkout, not another.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
