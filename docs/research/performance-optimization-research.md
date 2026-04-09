# Performance Optimization Research — vllm-mlx on Apple Silicon

**Date:** 2026-04-08
**Hardware:** Mac Studio, 512GB unified memory
**Goal:** Maximize tokens/sec without sacrificing quality

## Current Bottleneck Analysis

### The fundamental constraint on Apple Silicon

Token generation is **memory-bandwidth bound**, not compute-bound. The GPU needs to read every model weight once per token. With 400GB/s bandwidth on M4 Ultra and a 14GB model (4-bit 27B), the theoretical max is ~28 tok/s just from weight loading. Everything else (KV cache reads, compute, Python overhead) subtracts from that ceiling.

Prefill (TTFT) is **compute-bound** — large matrix multiplications benefit from GPU parallelism and M5's Neural Accelerators.

### Bottlenecks found in our codebase

| Area | File | Issue | Impact |
|------|------|-------|--------|
| Simple engine sync | `engine/simple.py:716,743` | Synchronous GPU evaluation blocks CPU after every token | 5-10% throughput loss |
| No graph compilation | All model forward passes | Each token rebuilds the compute graph from Python | 15-30% overhead on small models |
| Tokenization blocking | `engine/simple.py:726` | `tokenizer.decode()` runs inline in generation loop | 2-5% throughput loss |
| Serial prefill/generation | `scheduler.py:241-535` | New requests wait for current batch to finish prefilling | 15-25% throughput loss under load |
| Reactive memory flush | `engine_core.py:196-207` | Full `mx.clear_cache()` every 64 steps | Unpredictable latency spikes |
| No draft model speculation | N/A | Only MTP exists, not full draft-model speculative decoding | Missing 1.5-2.5x potential |

## Optimization Ideas — Ranked by Impact

### Tier 1: High impact, buildable now

#### 1. `mx.compile()` on model forward pass
**What:** MLX's `mx.compile()` traces and fuses the computation graph — merging elementwise ops (GELU, LayerNorm, RoPE, attention score computation) into single Metal kernels. Reduces kernel launch overhead and memory traffic.
**Evidence:** MLX docs show GELU going from bandwidth-bound to compute-bound after compilation. Apple's examples demonstrate significant speedups on transformer forward passes.
**Estimated gain:** 15-30% on prefill, 5-15% on decode (model-dependent)
**Effort:** Medium — wrap model `__call__` with `mx.compile()`, handle cache state correctly (compile can't capture mutable state)
**Risk:** Shape changes between tokens trigger recompilation. Need to handle variable sequence lengths.
**Reference:** https://ml-explore.github.io/mlx/build/html/usage/compile.html

#### 2. Async GPU evaluation in simple engine
**What:** Replace synchronous GPU evaluation with asynchronous evaluation so CPU work (tokenization, SSE framing, Python loop) overlaps with GPU computation.
**Evidence:** The batched scheduler already does this (`scheduler.py:200-202`). Simple engine doesn't.
**Estimated gain:** 5-10%
**Effort:** Low — change 4 lines in `engine/simple.py`
**Risk:** Minimal. Need to ensure token is ready before decode.
**Reference:** Already implemented in our scheduler code.

#### 3. `mx.export_function()` for precompiled models
**What:** Export compiled compute graphs to disk. First run traces and compiles; subsequent loads are instant. Eliminates JIT overhead on startup and first request.
**Evidence:** MLX 0.31+ supports this via `mx.export_function()` / `mx.import_function()`.
**Estimated gain:** Faster cold start (seconds to milliseconds for first token). No steady-state throughput change.
**Effort:** Medium
**Reference:** https://ml-explore.github.io/mlx/build/html/usage/export.html

### Tier 2: Bigger effort, bigger payoff

#### 4. Speculative decoding with draft model
**What:** Small (1-3B) draft model generates N candidate tokens. Main model verifies all N in one forward pass. Accepted tokens are "free" — you get multiple tokens per main-model step.
**Evidence:** 1.5-2.5x speedup widely reported in literature. Works best when draft model is high-acceptance (same family, e.g., Qwen-0.5B drafting for Qwen-30B).
**Estimated gain:** 40-150% on generation (depends on acceptance rate)
**Effort:** High — need draft model loading, speculative loop, verification, rollback on mismatch
**Risk:** Draft model adds memory pressure. Acceptance rate varies by task. Code-gen and structured output have higher acceptance than creative writing.
**Reference:** vLLM GPU has this. Our MTP is a limited version (same model, not separate draft).

#### 5. Interleaved chunked prefill + generation
**What:** Allow generation to continue for active requests while new requests prefill in chunks between decode steps. Currently serial.
**Evidence:** vLLM GPU does this. Our scheduler has chunked prefill but doesn't admit new requests between chunks.
**Estimated gain:** 15-25% throughput under concurrent load
**Effort:** Medium-High — refactor `scheduler.py` chunk loop to also schedule waiting requests
**Risk:** Memory pressure from more concurrent KV caches. Need careful budget management.
**Reference:** `scheduler.py:354` already has "generation step between chunks" — extend to also admit waiting requests.

#### 6. SSD-backed KV cache persistence
**What:** Persist KV cache to NVMe between requests. Repeated system prompts skip prefill entirely.
**Evidence:** oMLX (https://github.com/jundot/omlx) reports TTFT dropping from 30-90s to 1-3s on long prefixes. With 512GB RAM we might not need SSD — could keep hot caches in memory.
**Estimated gain:** 10-50x TTFT improvement for repeated prompts
**Effort:** Medium — serialize KV cache state, hash-based lookup, invalidation strategy
**Risk:** Cache staleness, memory management complexity
**Reference:** oMLX implementation: https://github.com/jundot/omlx

### Tier 3: Exotic / model-level

#### 7. Custom Metal kernels via `mx.fast.metal_kernel()`
**What:** Hand-write Metal shaders for hot-path operations: flash attention, fused QKV, fused RoPE+attention.
**Evidence:** llama.cpp's Metal kernels are hand-tuned and beat MLX on prefill. MLX's `mx.fast` module exposes kernel authoring.
**Estimated gain:** 20-40% on prefill (where MLX is weakest vs llama.cpp)
**Effort:** Very high — Metal shader programming, profiling, per-architecture tuning
**Reference:** https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html

#### 8. MXFP4 quantization (Apple's native format)
**What:** Mixed-precision 4-bit format with hardware acceleration on M4/M5 Neural Accelerators. Higher quality than INT4 at same size.
**Evidence:** Apple showed GPT-OSS 20B in native MXFP4 on M5 with no quality loss. 4.06x TTFT speedup over M4.
**Estimated gain:** Better quality at same speed as INT4, or faster at same quality
**Effort:** High — need MXFP4 conversion tooling, may need MLX framework support
**Risk:** Only accelerated on M4+ chips with Neural Accelerators. M1/M2/M3 won't benefit.
**Reference:** https://machinelearning.apple.com/research/exploring-llms-mlx-m5

#### 9. Tensor parallelism across GPU dies
**What:** Split model layers across multiple GPU compute units. On Ultra chips, both dies work in parallel.
**Evidence:** MLX has `mx.distributed` and a tensor parallelism example.
**Estimated gain:** ~1.8x on Ultra chips (2 dies), marginal on non-Ultra
**Effort:** Medium — MLX handles the distributed ops, but model sharding logic needed
**Reference:** https://ml-explore.github.io/mlx/build/html/examples/tensor_parallelism.html

#### 10. Model surgery — pruning and distillation
**What:** Remove less-important attention heads or layers, or distill a large model into a smaller one trained on the large model's outputs. Produces a smaller, faster model with similar quality.
**Evidence:** Widely used in production. Redis reports 40-70% cost reduction with minimal quality loss via distillation.
**Estimated gain:** Variable — depends on how aggressively you prune
**Effort:** Very high (distillation requires training infrastructure)
**Risk:** Quality degradation on edge cases
**Reference:** https://redis.io/blog/model-distillation-llm-guide/

## Key Ecosystem Context

### MLX vs llama.cpp (as of April 2026)
- **Decode (tok/s):** MLX is 1.4-1.8x faster than raw llama.cpp on Apple Silicon
- **Prefill (TTFT):** llama.cpp is faster due to hand-tuned Metal kernels and flash attention
- **Memory:** MLX uses 7-13% less memory due to zero-copy unified memory
- **MoE models:** MLX advantage is largest (up to 3x vs Ollama/llama.cpp)
- **Long context (30K+):** llama.cpp with flash attention is ~50% faster than MLX

### MLX weaknesses to be aware of
- Prefill is the biggest weakness — full prefill strategy without sliding-window cache
- Floating-point nondeterminism (batch size changes can affect outputs)
- Metal 4 type checking is stricter — bfloat/half mismatches can crash on M4+

### vllm-mlx's position
- Only MLX server with continuous batching (3.4x throughput with 5 concurrent requests)
- Paged KV cache with prefix sharing
- 15+ tool parsers for agentic use
- Unique position: could combine MLX's decode speed advantage with smarter scheduling

## Sources

- [Apple ML Research — MLX on M5 Neural Accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [MLX Compilation Documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- [MLX Export Functions Documentation](https://ml-explore.github.io/mlx/build/html/usage/export.html)
- [MLX Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [MLX Tensor Parallelism Example](https://ml-explore.github.io/mlx/build/html/examples/tensor_parallelism.html)
- [MLX: The Next Inference Engine for Apple Silicon (yage.ai analysis)](https://yage.ai/share/mlx-apple-silicon-en-20260331.html)
- [oMLX — SSD KV Cache Persistence](https://github.com/jundot/omlx)
- [vLLM-MLX Paper (arXiv)](https://arxiv.org/html/2601.19139v2)
- [Model Distillation Guide (Redis)](https://redis.io/blog/model-distillation-llm-guide/)
- [What to Buy for Local LLMs — April 2026 (Julien Simon)](https://julsimon.medium.com/what-to-buy-for-local-llms-april-2026-a4946a381a6a)
- [Ollama MLX Backend Announcement](https://ollama.com/blog/mlx)
- [MLX vs llama.cpp Comparison (Groundy)](https://groundy.com/articles/mlx-vs-llamacpp-on-apple-silicon-which-runtime-to-use-for-local-llm-inference/)

## What to build first

**Recommended sequence:**
1. Async GPU evaluation in simple engine (1-2 hours, easy win)
2. `mx.compile()` on forward pass (1-2 days, biggest single improvement)
3. Benchmark infrastructure (needed before anything else to measure real impact)
4. Speculative decoding with draft model (1-2 weeks, biggest total improvement)
5. Interleaved scheduling (1 week, best for multi-user scenarios)
