# Benchmarks de LLM

## Ejecutar benchmarks de LLM

```bash
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --prompts 5 --max-tokens 256
```

## Resultados (M4 Max, 128GB)

| Model | Gen Speed | TTFT* | Memory |
|-------|-----------|-------|--------|
| Qwen3-0.6B-8bit | 402.3 tok/s | 58.6 ms | 0.68 GB |
| Llama-3.2-1B-Instruct-4bit | 463.6 tok/s | 49.2 ms | 0.69 GB |
| Qwen2.5-1.5B-Instruct-4bit | 308.5 tok/s | 86.2 ms | 0.84 GB |
| Llama-3.2-3B-Instruct-4bit | 200.1 tok/s | 81.4 ms | 1.79 GB |
| Qwen3-30B-A3B-4bit | 123.9 tok/s | 126.9 ms | 16.05 GB |
| NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-6Bit | 122.9 tok/s | 72.3 ms | 23.98 GB |

*TTFT = Time to First Token (latencia hasta que el modelo comienza a generar)

## Resultados (M1 Max, 64GB)

| Model | Runs | Prompt Tok | Gen Tok | Total Time (s) | TTFT Mean (ms) | TPOT Mean (ms) | Gen Speed (tok/s) | Total Throughput (tok/s) |
|-------|------|------------|---------|-----------------|-----------------|-----------------|-------------------|--------------------------|
| Qwen3-0.6B-8bit | 5 | 56 | 1280 | 5.66 | 119.0 | 3.97 | 251.9 | 236.1 |

## Resultados de continuous batching

| Model | Single Request | Batch (5 req) | Speedup |
|-------|----------------|---------------|---------|
| Llama-3.2-1B-Instruct-4bit | 299.1 tok/s | 613.0 tok/s | **2.05x** |
| Llama-3.2-3B-Instruct-4bit | 137.6 tok/s | 208.1 tok/s | **1.51x** |
| Qwen3-0.6B-8bit | 328.1 tok/s | 1111.8 tok/s | **3.39x** |
| Qwen3-30B-A3B-4bit | 98.1 tok/s | 233.3 tok/s | **2.38x** |
| Qwen2.5-1.5B-Instruct-4bit | 196.9 tok/s | 322.2 tok/s | **1.64x** |

*Con 5 solicitudes concurrentes se observa una mejora de throughput de 1.5x a 3x.*

### Continuous batching (M1 Max, 64GB)

| Requests | Total Tokens | Total Time (s) | Throughput (tok/s) | Requests/sec |
|----------|--------------|-----------------|--------------------|--------------|
| 5 | 315 | 0.64 | 492.5 | 7.82 |

## Rendimiento de streaming

| Model | TTFT | Generation Speed |
|-------|------|------------------|
| Llama-3.2-1B-Instruct-4bit | ~4.6ms | 218.9 tok/s |
| Llama-3.2-3B-Instruct-4bit | ~10.7ms | 93.6 tok/s |
| Qwen3-0.6B-8bit | ~3.0ms | 328.5 tok/s |
| Qwen3-30B-A3B-4bit | ~10.2ms | 98.4 tok/s |
| Qwen2.5-1.5B-Instruct-4bit | ~7.1ms | 140.3 tok/s |

### Detokenizador en streaming (M1 Max, 64GB)

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

Speedup promedio: 1.79x

## Resultados del prefix cache

### Prefix cache (M4 Max, 128GB)

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

### Prefix cache (M1 Max, 64GB)

| Test | Expected | Actual | Time | Status |
|------|----------|--------|------|--------|
| First request | MISS | MISS | 203.5ms | PASS |
| Same prompt | HIT | HIT | 131.6ms | PASS |
| Different prompt | MISS or PREFIX_HIT | PREFIX_HIT (5 tok) | 135.3ms | PASS |

Estadísticas finales del cache:

| Cache Hits | Cache Misses | Hit Rate | Tokens Saved | Cached Speedup |
|------------|--------------|----------|--------------|----------------|
| 2 | 1 | 66.7% | 20 | 1.55x |

## Resultados del paged cache

*Prueba: 20 solicitudes de inferencia reales en 2 rondas con un system prompt compartido de aproximadamente 286 tokens*

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

### Paged KV cache (M1 Max, 64GB)

Benchmark de inferencia (20 solicitudes):

| Mode | Time (s) | Throughput (tok/s) |
|------|----------|--------------------|
| Without paged cache | 3.43 | 291.8 |
| With paged cache | 3.42 | 292.2 |

| Speedup | Blocks Allocated | Shared Blocks | Cache Hits | Tokens Saved |
|---------|------------------|---------------|------------|--------------|
| 1.00x | 45 | 4 | 10 | 2560 |

Inferencia concurrente real (20 solicitudes):

| Mode | Time (s) | Throughput (tok/s) |
|------|----------|--------------------|
| Without paged cache | 4.32 | 231.7 |
| With paged cache | 4.35 | 229.7 |

| Speedup | Blocks Allocated | Shared Blocks | Cache Hits | Tokens Saved |
|---------|------------------|---------------|------------|--------------|
| 0.99x | 49 | 8 | 10 | 5120 |

Demostración de ahorro de memoria:

| Scenario | Memory Savings |
|----------|----------------|
| Shared system prompts | 70.8% |
| Concurrent memory efficiency | 83.5% |
| Prefix sharing branches | 38.5% |

## Análisis del detokenizador en streaming

*Investigación Fase 9.1: `BPEStreamingDetokenizer` de mlx-lm vs `tokenizer.decode()` naive*

### Contexto

El enfoque naive llama a `decode([token])` por cada token. En teoria, los detokenizadores en streaming ofrecen complejidad O(T) frente a O(T²) del decode naive.

### Resultados del benchmark aislado

```bash
vllm-mlx bench-detok
```

Al reutilizar la misma instancia del detokenizador (con `reset()` entre usos):

| Sequence | Tokens | Naive decode() | Streaming | Speedup |
|----------|--------|----------------|-----------|---------|
| Short | 8 | 0.020ms | 0.019ms | 1.05x |
| Medium | 103 | 0.155ms | 0.097ms | 1.59x |
| Long | 511 | 0.752ms | 0.371ms | **2.03x** |
| 1K tokens | 1191 | 1.743ms | 0.833ms | **2.09x** |
| 2K tokens | 2381 | 3.493ms | 1.737ms | **2.01x** |

### Hallazgo clave: costo de creación de instancias

Crear una nueva instancia de `BPEStreamingDetokenizer` es **extremadamente costoso**:

```
100 tokenizer.detokenizer calls: 5.266s (52.7ms each!)
```

Esto significa que crear un nuevo detokenizador por solicitud agrega **aproximadamente 52ms de sobrecarga**, anulando cualquier beneficio.

### Impacto en uso real

Al integrarlo en el scheduler (un detokenizador por solicitud):

| Metric | Naive decode() | Streaming (new instance) |
|--------|----------------|--------------------------|
| Throughput (20 req) | 681 tok/s | 275 tok/s |
| Impact | - | **-60% slower** |

### Conclusión

El detokenizador en streaming **no es viable actualmente** para uso por solicitud, debido al costo de creación de instancias. El enfoque naive con `decode([token])` sigue siendo más rápido en la práctica.

**Optimizacion futura**: crear un pool de instancias de detokenizador al inicio y reutilizarlas entre solicitudes.

## Referencia de métricas

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to First Token: latencia hasta que el modelo comienza a responder (ms) |
| **TPOT** | Time Per Output Token: tiempo entre cada token generado (ms/token) |
| **Generation TPS** | Tokens de salida por segundo (tok/s) |
| **Processing TPS** | Tokens de entrada/prompt procesados por segundo (tok/s) |
| **End-to-End Latency** | Tiempo total desde la solicitud hasta la respuesta completa |
| **Total Throughput** | Tokens totales (entrada + salida) por segundo |

## Ejecutar benchmarks

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
