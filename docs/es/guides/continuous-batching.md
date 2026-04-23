# Continuous Batching

El continuous batching permite mayor throughput al servir múltiples usuarios concurrentes.

## Activar Continuous Batching

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --continuous-batching
```

## Con Paged Cache

Para compartir prefijos de forma eficiente en memoria:

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --continuous-batching --use-paged-cache
```

## Cómo Funciona

### Modo Simple (Predeterminado)
- Una solicitud a la vez
- Máximo throughput para un solo usuario
- Sin sobrecarga por batching

### Modo Continuous Batching
- Múltiples solicitudes procesadas en conjunto
- Mejor throughput para usuarios concurrentes
- Pequeña sobrecarga por solicitud

### Paged Cache
- KV cache almacenado en bloques de tamaño fijo
- Los system prompts compartidos usan los mismos bloques
- Ahorro de memoria: 80% o más con 10 o más usuarios concurrentes

## Resultados de Rendimiento

**Resultados de Continuous Batching (M4 Max, 128GB):**

| Modelo | Solicitud Individual | Batch (5 req) | Mejora |
|--------|----------------------|---------------|--------|
| Llama-3.2-1B-Instruct-4bit | 299.1 tok/s | 613.0 tok/s | **2.05x** |
| Llama-3.2-3B-Instruct-4bit | 137.6 tok/s | 208.1 tok/s | **1.51x** |
| Qwen3-0.6B-8bit | 328.1 tok/s | 1111.8 tok/s | **3.39x** |
| Qwen3-30B-A3B-4bit | 98.1 tok/s | 233.3 tok/s | **2.38x** |
| Qwen2.5-1.5B-Instruct-4bit | 196.9 tok/s | 322.2 tok/s | **1.64x** |

*El batching de 5 solicitudes concurrentes muestra una mejora de throughput de 1.5 a 3 veces.*

## Rendimiento en Streaming

**Rendimiento de Streaming (M4 Max, 128GB):**

| Modelo | TTFT | Velocidad de Generación |
|--------|------|-------------------------|
| Llama-3.2-1B-Instruct-4bit | ~4.6ms | 218.9 tok/s |
| Llama-3.2-3B-Instruct-4bit | ~10.7ms | 93.6 tok/s |
| Qwen3-0.6B-8bit | ~3.0ms | 328.5 tok/s |
| Qwen3-30B-A3B-4bit | ~10.2ms | 98.4 tok/s |
| Qwen2.5-1.5B-Instruct-4bit | ~7.1ms | 140.3 tok/s |

*TTFT = Time to First Token*

## Configuración de Streaming

Controla la entrega de tokens con `--stream-interval`:

```bash
# Cada token (más fluido)
vllm-mlx serve model --continuous-batching --stream-interval 1

# Tokens en batch (mejor para alta latencia)
vllm-mlx serve model --continuous-batching --stream-interval 5
```

| Valor | Comportamiento |
|-------|----------------|
| `1` | Envía cada token de inmediato |
| `2-5` | Agrupa tokens antes de enviar |
| `10+` | Máximo throughput, salida en fragmentos más grandes |

## Gestión de Memoria

En modelos grandes, el prefix cache puede consumir una cantidad significativa de memoria. El cache con gestión automática de memoria administra esto de forma transparente:

```bash
# Detección automática (usa el 20% de la RAM disponible)
vllm-mlx serve model --continuous-batching

# Límite explícito
vllm-mlx serve model --continuous-batching --cache-memory-mb 2048

# Porcentaje personalizado
vllm-mlx serve model --continuous-batching --cache-memory-percent 0.10
```

| Opción | Descripción |
|--------|-------------|
| `--cache-memory-mb` | Establece un límite explícito en MB |
| `--cache-memory-percent` | Fracción de la RAM disponible (predeterminado: 0.20) |
| `--no-memory-aware-cache` | Usa el cache heredado basado en conteo de entradas |

## Prefix Cache

El prefix caching reutiliza el KV cache para prompts repetidos.

### Cómo Funciona

```
User 1: System prompt (500 tokens) → Creates 8 blocks
User 2: Same system prompt → Shares 8 blocks (ref_count++)
User N: Same system prompt → Shares 8 blocks (ref_count++)

Memory savings: 80%+ for 10+ concurrent users
```

### Estrategia de Clave de Cache

- **LLM**: `hash(prompt)`
- **Images**: `hash(image_content) + hash(prompt)`
- **Videos**: `hash(video_path) + hash(fps) + hash(max_frames) + hash(prompt)`

### Probar el Prefix Cache

```bash
python tests/test_prefix_cache.py
```

```
======================================================================
  LLM PREFIX CACHE TEST
======================================================================
  Model: mlx-community/Qwen3-0.6B-8bit
  Expected behavior:
    - Same prompt → cache HIT
    - Different prompt → cache MISS or PREFIX_HIT (shared template tokens)
----------------------------------------------------------------------
  Results:
  Step   | Description         | Expected | Actual | Status
  -------+---------------------+----------+--------+-------
  1a     | First request       | MISS     | MISS   | PASS
  1b     | Same prompt         | HIT      | HIT    | PASS
  1c     | Different prompt    | MISS     | MISS   | PASS
  1d     | Return to prompt 1  | HIT      | HIT    | PASS
======================================================================
```

## Ejecutar Benchmarks

```bash
# Benchmark de continuous batching
python tests/test_continuous_batching.py

# Prueba de prefix cache
python tests/test_prefix_cache.py
```

## Cuándo Usarlo

| Escenario | Modo |
|-----------|------|
| Usuario individual, máxima velocidad | Simple (predeterminado) |
| Múltiples usuarios concurrentes | `--continuous-batching` |
| Modelos grandes (7B+) | `--continuous-batching --cache-memory-mb 2048` |
| Producción con prompts compartidos | `--continuous-batching --use-paged-cache` |

## Configuración para Producción

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --port 8000
```
