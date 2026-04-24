# Benchmarks

Performance benchmarks for vllm-mlx on Apple Silicon.

## Benchmark Types

- [LLM Benchmarks](llm.md) - Text generation performance
- [Image Benchmarks](image.md) - Image understanding performance
- [Video Benchmarks](video.md) - Video understanding performance

## Quick Commands

```bash
# LLM benchmark
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit

# Image benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Video benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --video

# Running-server prompt sweep with Prometheus metric deltas
vllm-mlx bench-serve --url http://localhost:8000 --prompts short,long \
  --concurrency 1,4 --output bench.json --format json

# Running-server product-style workload with quality checks
vllm-mlx bench-serve --url http://localhost:8000 \
  --workload ./workload.json --output workload-results.json
```

## Contract Workloads

`vllm-mlx bench-serve --workload` runs declarative cases against an already
running OpenAI-compatible server. This is intended for model and feature-stack
qualification, where raw speed is not enough and every run needs provenance,
quality checks, Prometheus metric deltas, and policy-timeout evidence.
Use `--repetitions` to measure variance; workload summaries report per-case
sample counts, failure rates, and min/median/max latency and throughput.

Example workload:

```json
{
  "name": "writing-contract",
  "description": "Representative long-form writing requests",
  "defaults": {
    "max_tokens": 32768,
    "enable_thinking": true,
    "policy_timeout_ms": 180000,
    "checks": {
      "finish_reason": "stop",
      "forbidden_regex": ["<think>", "prompt leakage"],
      "min_chars": 500
    }
  },
  "cases": [
    {
      "id": "resume-golden-1",
      "messages": [
        {"role": "user", "content": "Write the requested artifact..."}
      ],
      "tags": ["resume", "quality-floor"]
    }
  ]
}
```

Cases can also reference an existing OpenAI-compatible request JSON instead of
duplicating a large prompt body:

```json
{
  "name": "writing-contract",
  "cases": [
    {
      "id": "resume-golden-1",
      "request_path": "./fixtures/job543_resume_precise_request.json",
      "checks": {
        "finish_reason": "stop",
        "forbidden_regex": ["<think>"]
      }
    }
  ]
}
```

When `request_path` is used, `messages`, `max_tokens`, `enable_thinking`, and
extra request-body fields such as `thinking_token_budget` are read from that
file. Case-level `extra_body` values override request-file values.

`policy_timeout_ms` is recorded as comparison evidence. It is not treated as a
hardware capability claim. Use it to answer "would this run fit my product
policy?" after first measuring what the model and serving stack can actually do.

Workload output defaults to JSON for full provenance. Use `--format csv` for
flat per-case rows or `--format sql` to emit importable SQL for a local
benchmark database.

```bash
vllm-mlx bench-serve --url http://localhost:8000 \
  --workload ./workload.json --repetitions 5 --output workload-results.json
```

## Standalone Test Defaults

Standalone benchmark test scripts have built-in default models, so you can run:

```bash
python tests/test_continuous_batching.py
python tests/test_prefix_cache.py
```

Defaults:
- `tests/test_continuous_batching.py` → `mlx-community/Qwen3-8B-6bit`
- `tests/test_prefix_cache.py` → `mlx-community/Qwen3-0.6B-8bit`

To test different models, use the optional `--model` flag:

```bash
python tests/test_continuous_batching.py --model mlx-community/Qwen3-0.6B-8bit
python tests/test_prefix_cache.py --model mlx-community/Qwen3-8B-6bit
```

## Hardware

Benchmarks have been collected on the following Apple Silicon configurations:

| Chip | Memory | Python |
|------|--------|--------|
| Apple M4 Max | 128 GB unified | 3.13 |
| Apple M1 Max | 64 GB unified | 3.12 |

Results will vary on different Apple Silicon chips.

## Contributing Benchmarks

If you have a different Apple Silicon chip, please share your results:

```bash
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit --output results.json
```

Open an issue with your results at [GitHub Issues](https://github.com/waybarrios/vllm-mlx/issues).
