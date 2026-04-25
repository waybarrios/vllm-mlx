# Référence CLI

## Vue d'ensemble des commandes

| Commande | Description |
|---------|-------------|
| `vllm-mlx serve` | Démarrer le serveur compatible OpenAI |
| `vllm-mlx-bench` | Exécuter des benchmarks de performance |
| `vllm-mlx-chat` | Démarrer l'interface de chat Gradio |

## `vllm-mlx serve`

Démarrer le serveur API compatible OpenAI.

### Utilisation

```bash
vllm-mlx serve <model> [options]
```

### Options

| Option | Description | Défaut |
|--------|-------------|---------|
| `--served-model-name` | Nom de modèle personnalisé exposé via l'API OpenAI. Si non défini, le chemin du modèle est utilisé comme nom. | None |
| `--port` | Port du serveur | 8000 |
| `--host` | Hôte du serveur | 127.0.0.1 |
| `--api-key` | Clé API pour l'authentification | None |
| `--rate-limit` | Requêtes par minute par client (0 = désactivé) | 0 |
| `--timeout` | Délai d'attente des requêtes en secondes | 300 |
| `--enable-metrics` | Exposer les métriques Prometheus sur `/metrics` | False |
| `--continuous-batching` | Activer le continuous batching pour plusieurs utilisateurs | False |
| `--cache-memory-mb` | Limite mémoire du cache en Mo | Auto |
| `--cache-memory-percent` | Fraction de la RAM réservée au cache | 0.20 |
| `--no-memory-aware-cache` | Utiliser le cache legacy basé sur le nombre d'entrées | False |
| `--use-paged-cache` | Activer le KV cache paginé | False |
| `--max-tokens` | Nombre maximum de tokens par défaut | 32768 |
| `--max-request-tokens` | Valeur maximale de `max_tokens` acceptée depuis les clients API | 32768 |
| `--stream-interval` | Tokens par fragment de streaming | 1 |
| `--mcp-config` | Chemin vers le fichier de configuration MCP | None |
| `--paged-cache-block-size` | Tokens par bloc de cache | 64 |
| `--max-cache-blocks` | Nombre maximum de blocs de cache | 1000 |
| `--max-num-seqs` | Nombre maximum de séquences simultanées | 256 |
| `--default-temperature` | Température par défaut si non spécifiée dans la requête | None |
| `--default-top-p` | Valeur top_p par défaut si non spécifiée dans la requête | None |
| `--max-audio-upload-mb` | Taille maximale des fichiers audio téléversés pour `/v1/audio/transcriptions` | 25 |
| `--max-tts-input-chars` | Longueur maximale du texte acceptée par `/v1/audio/speech` | 4096 |
| `--reasoning-parser` | Analyseur pour les modèles de reasoning (`qwen3`, `deepseek_r1`) | None |
| `--embedding-model` | Pré-charger un modèle d'embeddings au démarrage | None |
| `--enable-auto-tool-choice` | Activer le tool calling automatique | False |
| `--tool-call-parser` | Analyseur d'appels d'outils (`auto`, `mistral`, `qwen`, `llama`, `hermes`, `deepseek`, `kimi`, `granite`, `nemotron`, `xlam`, `functionary`, `glm47`) | None |

### Exemples

```bash
# Simple mode (single user, max throughput)
# Model path is used as the model name in the OpenAI API (e.g. model="mlx-community/Llama-3.2-3B-Instruct-4bit")
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit

Model will show up as 'mlx-community/Llama-3.2-3B-Instruct-4bit' in the `/v1/models` API endpoint. View with `curl http://localhost:8000/v1/models` or similar.

# With a custom API model name (model is accessed as "my-model" via the OpenAI API)
# --served-model-name sets the name clients must use when calling the API (e.g. model="my-model")
vllm-mlx serve --served-model-name my-model mlx-community/Llama-3.2-3B-Instruct-4bit
# Note: Model will show up as 'my-model' in the `/v1/models` API endpoint.

# Continuous batching (multiple users)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

# With memory limit for large models
vllm-mlx serve mlx-community/GLM-4.7-Flash-4bit \
  --continuous-batching \
  --cache-memory-mb 2048

# Production with paged cache
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --port 8000

# With MCP tools
vllm-mlx serve mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

# Multimodal model
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit

# Reasoning model (separates thinking from answer)
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# DeepSeek reasoning model
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1

# Tool calling with Mistral/Devstral
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral

# Tool calling with Granite
vllm-mlx serve mlx-community/granite-4.0-tiny-preview-4bit \
  --enable-auto-tool-choice --tool-call-parser granite

# With API key authentication
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --api-key your-secret-key

# Expose Prometheus metrics
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --enable-metrics

# Production setup with security options
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120 \
  --continuous-batching
```

### Sécurité

Lorsque `--api-key` est défini, toutes les requêtes API requièrent l'en-tête `Authorization: Bearer <api-key>` :

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key"  # Must match --api-key
)
```

Ou avec curl :

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-secret-key"
```

## `vllm-mlx-bench`

Exécuter des benchmarks de performance.

### Utilisation

```bash
vllm-mlx-bench --model <model> [options]
```

### Options

| Option | Description | Défaut |
|--------|-------------|---------|
| `--model` | Nom du modèle | Requis |
| `--prompts` | Nombre de prompts | 5 |
| `--max-tokens` | Nombre maximum de tokens par prompt | 256 |
| `--quick` | Mode benchmark rapide | False |
| `--video` | Exécuter le benchmark vidéo | False |
| `--video-url` | URL vidéo personnalisée | None |
| `--video-path` | Chemin vidéo personnalisé | None |

### Exemples

```bash
# LLM benchmark
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit

# Quick benchmark
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --quick

# Image benchmark (auto-detected for VLM models)
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Video benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --video

# Custom video
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit \
  --video --video-url https://example.com/video.mp4
```

## `vllm-mlx-chat`

Démarrer l'interface de chat Gradio.

### Utilisation

```bash
vllm-mlx-chat --served-model-name <model-name> [options]
```

### Options

| Option | Description | Défaut |
|--------|-------------|---------|
| `--model` | Nom du modèle | Requis |
| `--port` | Port Gradio | 7860 |
| `--text-only` | Désactiver le multimodal | False |

### Exemples

```bash
# Multimodal chat (text + images + video)
vllm-mlx-chat --served-model-name mlx-community/Qwen3-VL-4B-Instruct-3bit

# Text-only chat
vllm-mlx-chat --served-model-name mlx-community/Llama-3.2-3B-Instruct-4bit --text-only
```

## Variables d'environnement

| Variable | Description |
|----------|-------------|
| `VLLM_MLX_TEST_MODEL` | Modèle utilisé pour les tests |
| `HF_TOKEN` | Jeton HuggingFace |
