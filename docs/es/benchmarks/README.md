# Benchmarks

Benchmarks de rendimiento para vllm-mlx en Apple Silicon.

## Tipos de benchmark

- [Benchmarks de LLM](llm.md) - Rendimiento de generación de texto
- [Benchmarks de imagen](image.md) - Rendimiento de comprension de imagenes
- [Benchmarks de video](video.md) - Rendimiento de comprension de video

## Comandos rápidos

```bash
# LLM benchmark
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit

# Image benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Video benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --video
```

## Valores predeterminados de los scripts de prueba

Los scripts de benchmark independientes tienen modelos predeterminados integrados, por lo que puedes ejecutar:

```bash
python tests/test_continuous_batching.py
python tests/test_prefix_cache.py
```

Valores predeterminados:
- `tests/test_continuous_batching.py` → `mlx-community/Qwen3-8B-6bit`
- `tests/test_prefix_cache.py` → `mlx-community/Qwen3-0.6B-8bit`

Para probar con otros modelos, usa el parámetro opcional `--model`:

```bash
python tests/test_continuous_batching.py --model mlx-community/Qwen3-0.6B-8bit
python tests/test_prefix_cache.py --model mlx-community/Qwen3-8B-6bit
```

## Hardware

Los benchmarks se recopilaron en las siguientes configuraciones de Apple Silicon:

| Chip | Memoria | Python |
|------|---------|--------|
| Apple M4 Max | 128 GB unificada | 3.13 |
| Apple M1 Max | 64 GB unificada | 3.12 |

Los resultados pueden variar en distintos chips de Apple Silicon.

## Contribuir benchmarks

Si tienes un chip de Apple Silicon diferente, comparte tus resultados:

```bash
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit --output results.json
```

Abre un issue con tus resultados en [GitHub Issues](https://github.com/waybarrios/vllm-mlx/issues).
