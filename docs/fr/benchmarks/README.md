# Benchmarks

Benchmarks de performance pour vllm-mlx sur Apple Silicon.

## Types de benchmarks

- [Benchmarks LLM](llm.md) - Performance de génération de texte
- [Benchmarks image](image.md) - Performance de compréhension d'images
- [Benchmarks vidéo](video.md) - Performance de compréhension de vidéos

## Commandes rapides

```bash
# LLM benchmark
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit

# Image benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Video benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --video
```

## Valeurs par défaut des scripts de test autonomes

Les scripts de benchmark autonomes disposent de modèles par défaut intégrés, ce qui permet de les lancer directement :

```bash
python tests/test_continuous_batching.py
python tests/test_prefix_cache.py
```

Valeurs par défaut :
- `tests/test_continuous_batching.py` → `mlx-community/Qwen3-8B-6bit`
- `tests/test_prefix_cache.py` → `mlx-community/Qwen3-0.6B-8bit`

Pour tester d'autres modèles, utilisez l'option `--model` :

```bash
python tests/test_continuous_batching.py --model mlx-community/Qwen3-0.6B-8bit
python tests/test_prefix_cache.py --model mlx-community/Qwen3-8B-6bit
```

## Matériel

Les benchmarks ont été collectés sur les configurations Apple Silicon suivantes :

| Puce | Mémoire | Python |
|------|---------|--------|
| Apple M4 Max | 128 Go unifiée | 3.13 |
| Apple M1 Max | 64 Go unifiée | 3.12 |

Les résultats varieront selon la puce Apple Silicon utilisée.

## Contribuer des benchmarks

Si vous disposez d'une puce Apple Silicon différente, partagez vos résultats :

```bash
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit --output results.json
```

Ouvrez un ticket avec vos résultats sur [GitHub Issues](https://github.com/waybarrios/vllm-mlx/issues).
