# Capacity-envelope benchmarking

`bench-serve --capacity-envelope` measures how much concurrent work a running
OpenAI-compatible server can sustain under explicit latency and failure SLOs.
It is an HTTP-only client: it does not load a model or change serving defaults.

```bash
vllm-mlx bench-serve --url http://localhost:8000 \
  --capacity-envelope \
  --capacity-concurrency 1,2,4,8,16 \
  --capacity-prompt-tokens 256,2048 \
  --capacity-output-tokens 128 \
  --capacity-cache-modes cold,warm,prefix-hit,prefix-miss \
  --capacity-max-p95-ttft-ms 1000 \
  --capacity-max-p95-e2e-ms 10000 \
  --output capacity.json
```

The versioned JSON artifact includes the canonical workload hash, runtime and
host provenance, request samples, p50/p95/p99 cell summaries, successful-token
throughput, request failures (including timeout and OOM classifications), cache
actions, optional server telemetry, sustainable concurrency, and throughput per
GiB. Missing telemetry is `null`/unavailable; it is never estimated as zero.

## Measurement rules

- A cell fails when any request exceeds the configured failure allowance.
- Throughput uses successful completion tokens and measured batch wall time.
- A response that reaches the requested output limit is successful. A response
  without authoritative OpenAI `usage` token counts fails instead of producing
  zero-token throughput.
- `chunk_gap_ms` measures content-SSE chunk gaps because an HTTP chunk is not
  necessarily one token. Aggregate token throughput uses server `usage` counts.
- Every cache case starts with a successful `DELETE /v1/cache`; the run aborts
  if the server cannot verify that reset. Cold cells do not run a warmup request.
- Warm cells exercise model-warm/no-prefix-hit behavior by warming a distinct
  prompt. Prefix-hit cells warm the measured prompt; prefix-miss cells warm a
  deterministic alternate prefix before measuring the requested prompt.
- Prefix-hit/miss labels must be corroborated by server cache counters. Servers
  without an engine cache report those cells unavailable rather than aborting.
- Memory and optional MTP/speculative/cache counters include their observation
  source. Unsupported or inaccessible endpoints remain unavailable.
- Optional A/B parity uses greedy generation only. It compares token IDs when
  the server emits them and otherwise compares a decoded-output SHA-256 hash.

Prompt-token targets are workload buckets, not tokenizer claims. Each sample
records the server-reported prompt-token count as the authoritative observation.
Only compare artifacts whose workload and provenance describe compatible model,
quantization, software, cache, sampling, and hardware conditions.
