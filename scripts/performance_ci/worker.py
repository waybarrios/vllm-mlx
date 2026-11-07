"""One fresh-process, fixed-length native workload against one source tree."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def _hash(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def make_prompts(tokenizer, length, concurrency):
    seed = tokenizer.encode(
        "The quick brown fox studies reproducible performance measurements. ",
        add_special_tokens=False,
    )
    suffixes = list(dict.fromkeys(seed))
    if length < 2 or concurrency < 1 or concurrency > len(suffixes):
        raise ValueError("Cannot construct distinct fixed-length prompts")
    prefix = (seed * ((length // len(seed)) + 1))[: length - 1]
    return [prefix + [suffixes[i]] for i in range(concurrency)]


def eos_processor(token_ids):
    """Benchmark-only EOS suppression, leaving all other logits unchanged."""
    import mlx.core as mx

    masks = {}

    def process(tokens, logits):
        key = (logits.shape[-1], logits.dtype)
        if key not in masks:
            mask = mx.zeros((logits.shape[-1],), dtype=logits.dtype)
            if token_ids:
                mask[mx.array(sorted(token_ids))] = -float("inf")
            masks[key] = mask
        return logits + masks[key]

    return process


async def measure(loader, case):
    """Load and step on the same owner thread; time two complete batches."""
    import mlx.core as mx
    from vllm_mlx.engine_core import EngineConfig, EngineCore
    from vllm_mlx.request import SamplingParams
    from vllm_mlx.scheduler import SchedulerConfig
    from vllm_mlx.mlx_streams import bind_generation_streams

    loop = asyncio.get_running_loop()
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="performance-owner")
    engine = None

    async def on_owner(function):
        return await loop.run_in_executor(worker, function)

    def load():
        bind_generation_streams()
        mx.random.seed(0)
        started = time.perf_counter()
        model, tokenizer = loader()
        mx.eval(model.parameters())
        mx.synchronize()
        return model, tokenizer, time.perf_counter() - started

    try:
        model, tokenizer, load_seconds = await on_owner(load)
        prompts = make_prompts(tokenizer, case["prompt_tokens"], case["concurrency"])
        config = EngineConfig(
            model_name="performance-ci",
            scheduler_config=SchedulerConfig(
                max_num_seqs=case["concurrency"],
                prefill_batch_size=case["concurrency"],
                completion_batch_size=case["concurrency"],
                enable_prefix_cache=case["prefix_cache"],
                cache_memory_mb=128,
            ),
        )
        engine = await on_owner(
            lambda: EngineCore(model, tokenizer, config, generation_worker=worker)
        )
        stop_ids = await on_owner(engine.scheduler._get_stop_tokens)
        params = SamplingParams(
            max_tokens=case["output_tokens"],
            temperature=0,
            top_p=1,
            logits_processors=[eos_processor(stop_ids)],
        )
        await engine.start()
        phases = {}
        for phase in ("first", "warm"):
            await on_owner(mx.reset_peak_memory)
            started = time.perf_counter()

            async def request(index, prompt):
                admitted = time.perf_counter()
                rid = await engine.add_request(
                    prompt, params, request_id=f"{phase}-{index}"
                )
                req = engine.scheduler.get_request(rid)
                first = None
                final = None
                async for output in engine.stream_outputs(rid, timeout=60):
                    if first is None and output.new_token_ids:
                        first = time.perf_counter()
                    if output.finished:
                        final = output
                if (
                    first is None
                    or final is None
                    or final.finish_reason != "length"
                    or len(final.output_token_ids) != case["output_tokens"]
                ):
                    raise RuntimeError(f"Incomplete or short generation for {rid}")
                return {
                    "ttft": first - admitted,
                    "tokens": final.output_token_ids,
                    "cached": req.cached_tokens,
                }

            results = await asyncio.wait_for(
                asyncio.gather(*(request(i, p) for i, p in enumerate(prompts))),
                timeout=90,
            )
            await on_owner(mx.synchronize)
            elapsed = time.perf_counter() - started
            peak = await on_owner(mx.get_peak_memory)
            total = sum(len(result["tokens"]) for result in results)
            import statistics

            phases[phase] = {
                "batch_seconds": elapsed,
                "ttft_seconds": statistics.median(r["ttft"] for r in results),
                "tokens_per_second": total / elapsed,
                "peak_mlx_bytes": peak,
                "output_tokens": total,
                "output_sha256": _hash([r["tokens"] for r in results]),
                "cached_tokens": sum(r["cached"] for r in results),
            }
        return {
            "model_load_seconds": load_seconds,
            "prompt_sha256": _hash(prompts),
            "phases": phases,
        }
    finally:
        try:
            if engine is not None:
                await engine.stop()
                await on_owner(engine.close)
        finally:
            worker.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    import mlx.core as mx
    import vllm_mlx
    from mlx_lm import load

    if (
        platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or not mx.metal.is_available()
    ):
        raise RuntimeError("Expected an Apple Silicon GPU")
    mx.set_default_device(mx.gpu)
    if not Path(vllm_mlx.__file__).resolve().is_relative_to(source / "vllm_mlx"):
        raise RuntimeError("Imported the wrong source checkout")
    actual_sha = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_sha != spec["source_sha"]:
        raise RuntimeError("Source SHA mismatch")
    for name, version in spec["runtime_pins"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Runtime version mismatch for {name}")
    model_path = Path(spec["model_path"])
    if model_path.name != spec["case"]["revision"]:
        raise RuntimeError("Expected a model snapshot at the pinned revision")
    measurement = asyncio.run(measure(lambda: load(str(model_path)), spec["case"]))
    runtime = {
        "python": sys.version,
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
    }
    result = {
        "side": spec["side"],
        "repeat": spec["repeat"],
        "case": spec["case"],
        "source_sha": actual_sha,
        "runtime": runtime,
        "device": str(mx.default_device()),
        **measurement,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
