# LLM 基准测试

## 运行 LLM 基准测试

```bash
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompts 5 --max-tokens 256
```

## 测试结果 (M4 Max, 128GB)

| Model | Gen Speed | TTFT* | Memory |
|-------|-----------|-------|--------|
| Qwen3-0.6B-8bit | 402.3 tok/s | 58.6 ms | 0.68 GB |
| Llama-3.2-1B-Instruct-4bit | 463.6 tok/s | 49.2 ms | 0.69 GB |
| Qwen2.5-1.5B-Instruct-4bit | 308.5 tok/s | 86.2 ms | 0.84 GB |
| Llama-3.2-3B-Instruct-4bit | 200.1 tok/s | 81.4 ms | 1.79 GB |
| Qwen3-30B-A3B-4bit | 123.9 tok/s | 126.9 ms | 16.05 GB |
| NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-6Bit | 122.9 tok/s | 72.3 ms | 23.98 GB |

*TTFT = 首个 token 的生成时间（模型开始输出前的延迟）

## 测试结果 (M1 Max, 64GB)

| Model | Runs | Prompt Tok | Gen Tok | Total Time (s) | TTFT Mean (ms) | TPOT Mean (ms) | Gen Speed (tok/s) | Total Throughput (tok/s) |
|-------|------|------------|---------|-----------------|-----------------|-----------------|-------------------|--------------------------|
| Qwen3-0.6B-8bit | 5 | 56 | 1280 | 5.66 | 119.0 | 3.97 | 251.9 | 236.1 |

## Continuous Batching 测试结果

| Model | Single Request | Batch (5 req) | Speedup |
|-------|----------------|---------------|---------|
| Llama-3.2-1B-Instruct-4bit | 299.1 tok/s | 613.0 tok/s | **2.05x** |
| Llama-3.2-3B-Instruct-4bit | 137.6 tok/s | 208.1 tok/s | **1.51x** |
| Qwen3-0.6B-8bit | 328.1 tok/s | 1111.8 tok/s | **3.39x** |
| Qwen3-30B-A3B-4bit | 98.1 tok/s | 233.3 tok/s | **2.38x** |
| Qwen2.5-1.5B-Instruct-4bit | 196.9 tok/s | 322.2 tok/s | **1.64x** |

*批量处理 5 个并发请求可将 throughput 提升 1.5 到 3 倍。*

### Continuous Batching (M1 Max, 64GB)

| Requests | Total Tokens | Total Time (s) | Throughput (tok/s) | Requests/sec |
|----------|--------------|-----------------|--------------------|--------------|
| 5 | 315 | 0.64 | 492.5 | 7.82 |

## Streaming 性能

| Model | TTFT | Generation Speed |
|-------|------|------------------|
| Llama-3.2-1B-Instruct-4bit | ~4.6ms | 218.9 tok/s |
| Llama-3.2-3B-Instruct-4bit | ~10.7ms | 93.6 tok/s |
| Qwen3-0.6B-8bit | ~3.0ms | 328.5 tok/s |
| Qwen3-30B-A3B-4bit | ~10.2ms | 98.4 tok/s |
| Qwen2.5-1.5B-Instruct-4bit | ~7.1ms | 140.3 tok/s |

### Streaming 解码器 (M1 Max, 64GB)

`vllm-mlx bench-detok`:

| Tokens | Iterations | Naive Time | Streaming Time | Speedup |
|--------|------------|------------|----------------|---------|
| 742 | 5 | 1.69ms | 0.71ms | 2.39x |

`examples/benchmark_detokenizer.py`:

| Sequence | Tokens | decode() | Streaming | Speedup |
|----------|--------|----------|-----------|---------|
| Short | 8 | 0.029ms | 0.028ms | 1.04x |
| Medium | 103 | 0.206ms | 0.129ms | 1.59x |
| Long | 511 | 1.040ms | 0.502ms | 2.07x |
| 1K | 1191 | 2.446ms | 1.178ms | 2.08x |
| 2K | 2381 | 4.949ms | 2.356ms | 2.10x |
| 4K | 4761 | 9.887ms | 5.398ms | 1.83x |

平均加速比：1.79x

## Prefix Cache 测试结果

### Prefix Cache (M4 Max, 128GB)

```
======================================================================
  LLM PREFIX CACHE TEST
======================================================================
  Model: mlx-community/Qwen3-0.6B-8bit
  Expected behavior:
    - Same prompt → cache HIT
    - Different prompt → cache MISS
----------------------------------------------------------------------
  Results:
  Step   | Description         | Expected | Actual | Status
  -------+---------------------+----------+--------+-------
  1a     | First request       | MISS     | MISS   | ✓
  1b     | Same prompt         | HIT      | HIT    | ✓
  1c     | Different prompt    | MISS     | MISS   | ✓
  1d     | Return to prompt 1  | HIT      | HIT    | ✓
======================================================================
```

### Prefix Cache (M1 Max, 64GB)

| Test | Expected | Actual | Time | Status |
|------|----------|--------|------|--------|
| First request | MISS | MISS | 203.5ms | PASS |
| Same prompt | HIT | HIT | 131.6ms | PASS |
| Different prompt | MISS or PREFIX_HIT | PREFIX_HIT (5 tok) | 135.3ms | PASS |

最终缓存统计：

| Cache Hits | Cache Misses | Hit Rate | Tokens Saved | Cached Speedup |
|------------|--------------|----------|--------------|----------------|
| 2 | 1 | 66.7% | 20 | 1.55x |

## Paged Cache 测试结果

*测试：2 轮共 20 个真实推理请求，系统提示约 286 个 token*

```
======================================================================
  PAGED KV CACHE - REAL INFERENCE TEST
======================================================================

--------------------------------------------------
Test 1: WITHOUT Paged Cache (2 rounds of 10)
--------------------------------------------------
  Time: 1.47s
  Throughput: 681.2 tok/s
  Cache hits: 0
  Tokens saved: 0

--------------------------------------------------
Test 2: WITH Paged Cache (2 rounds of 10)
--------------------------------------------------
  Time: 1.31s
  Throughput: 765.8 tok/s

  Paged Cache Stats:
    Blocks allocated: 25
    Shared blocks: 4
    Cache hits: 10
    Tokens saved: 2560

==================================================
SUMMARY
==================================================
  Without paged cache: 681.2 tok/s
  With paged cache:    765.8 tok/s

  Speedup: 1.12x
  Cache hits: 10 (all Round 2 requests)
  Tokens saved: 2,560 (~256 tokens × 10 requests)
==================================================
```

### Paged KV Cache (M1 Max, 64GB)

推理基准测试（20 个请求）：

| Mode | Time (s) | Throughput (tok/s) |
|------|----------|--------------------|
| Without paged cache | 3.43 | 291.8 |
| With paged cache | 3.42 | 292.2 |

| Speedup | Blocks Allocated | Shared Blocks | Cache Hits | Tokens Saved |
|---------|------------------|---------------|------------|--------------|
| 1.00x | 45 | 4 | 10 | 2560 |

真实并发推理（20 个请求）：

| Mode | Time (s) | Throughput (tok/s) |
|------|----------|--------------------|
| Without paged cache | 4.32 | 231.7 |
| With paged cache | 4.35 | 229.7 |

| Speedup | Blocks Allocated | Shared Blocks | Cache Hits | Tokens Saved |
|---------|------------------|---------------|------------|--------------|
| 0.99x | 49 | 8 | 10 | 5120 |

内存节省示例：

| Scenario | Memory Savings |
|----------|----------------|
| Shared system prompts | 70.8% |
| Concurrent memory efficiency | 83.5% |
| Prefix sharing branches | 38.5% |

## Streaming 解码器分析

*第 9.1 阶段调查：mlx-lm 的 `BPEStreamingDetokenizer` 与朴素 `tokenizer.decode()` 对比*

### 背景

朴素方法对每个 token 调用 `decode([token])`。理论上，streaming 解码器的时间复杂度为 O(T)，而朴素解码为 O(T²)。

### 孤立基准测试结果

```bash
vllm-mlx bench-detok
```

复用同一解码器实例时（每次使用前调用 `reset()`）：

| Sequence | Tokens | Naive decode() | Streaming | Speedup |
|----------|--------|----------------|-----------|---------|
| Short | 8 | 0.020ms | 0.019ms | 1.05x |
| Medium | 103 | 0.155ms | 0.097ms | 1.59x |
| Long | 511 | 0.752ms | 0.371ms | **2.03x** |
| 1K tokens | 1191 | 1.743ms | 0.833ms | **2.09x** |
| 2K tokens | 2381 | 3.493ms | 1.737ms | **2.01x** |

### 关键发现：实例创建开销

创建新的 `BPEStreamingDetokenizer` 实例**极其昂贵**：

```
100 tokenizer.detokenizer calls: 5.266s (52.7ms each!)
```

这意味着每个请求新建一个解码器实例会增加约 **52ms 的额外开销**，从而抵消所有性能收益。

### 实际影响

集成到调度器后（每个请求一个解码器实例）：

| Metric | Naive decode() | Streaming (new instance) |
|--------|----------------|--------------------------|
| Throughput (20 req) | 681 tok/s | 275 tok/s |
| Impact | - | **慢 60%** |

### 结论

由于实例创建成本过高，streaming 解码器**目前不适合**在每个请求中独立使用。朴素的 `decode([token])` 方法在实践中仍然更快。

**未来优化方向**：在启动时预先创建一个解码器实例池，并在请求间复用这些实例。

## 指标参考

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to First Token，模型开始响应前的延迟 (ms) |
| **TPOT** | Time Per Output Token，每个生成 token 之间的间隔 (ms/token) |
| **Generation TPS** | 每秒输出 token 数 (tok/s) |
| **Processing TPS** | 每秒处理输入或提示词 token 数 (tok/s) |
| **End-to-End Latency** | 从请求发出到收到完整响应的总时间 |
| **Total Throughput** | 每秒处理的总 token 数（输入加输出） |

## 运行基准测试

```bash
# Basic benchmark
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit

# With more prompts
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit --prompts 10

# Save results
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit --output results.json

# Continuous batching test
python tests/test_continuous_batching.py

# Prefix cache test
python tests/test_prefix_cache.py

# Paged cache test
python tests/test_paged_cache_real_inference.py

# Streaming detokenizer benchmark
vllm-mlx bench-detok
vllm-mlx bench-detok mlx-community/Llama-3.2-1B-Instruct-4bit --iterations 5
```
