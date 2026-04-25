# Référence de configuration

## Configuration du serveur

### Options de base

| Option | Description | Défaut |
|--------|-------------|---------|
| `--host` | Adresse hôte du serveur | `127.0.0.1` |
| `--port` | Port du serveur | `8000` |
| `--max-tokens` | Nombre maximum de tokens par défaut | `32768` |
| `--max-request-tokens` | Valeur maximale de `max_tokens` acceptée depuis les clients API | `32768` |
| `--default-temperature` | Température par défaut si non spécifiée dans la requête | None |
| `--default-top-p` | top_p par défaut si non spécifié dans la requête | None |

### Options de sécurité

| Option | Description | Défaut |
|--------|-------------|---------|
| `--api-key` | Clé API pour l'authentification | None |
| `--rate-limit` | Requêtes par minute par client (0 = désactivé) | `0` |
| `--timeout` | Délai d'expiration des requêtes en secondes | `300` |
| `--enable-metrics` | Expose les métriques Prometheus sur `/metrics` | `false` |
| `--max-audio-upload-mb` | Taille maximale du fichier audio téléversé pour `/v1/audio/transcriptions` | `25` |
| `--max-tts-input-chars` | Longueur maximale du texte acceptée par `/v1/audio/speech` | `4096` |

### Options de batching

| Option | Description | Défaut |
|--------|-------------|---------|
| `--continuous-batching` | Active le continuous batching | `false` |
| `--stream-interval` | Tokens par fragment de streaming | `1` |
| `--max-num-seqs` | Nombre maximum de séquences simultanées | `256` |

### Options de cache

| Option | Description | Défaut |
|--------|-------------|---------|
| `--cache-memory-mb` | Limite mémoire du cache en Mo | Auto |
| `--cache-memory-percent` | Fraction de la RAM allouée au cache | `0.20` |
| `--no-memory-aware-cache` | Utilise le cache legacy basé sur le nombre d'entrées | `false` |
| `--use-paged-cache` | Active le KV cache paginé | `false` |
| `--paged-cache-block-size` | Tokens par bloc | `64` |
| `--max-cache-blocks` | Nombre maximum de blocs | `1000` |

### Options d'appel d'outils

| Option | Description | Défaut |
|--------|-------------|---------|
| `--enable-auto-tool-choice` | Active l'appel automatique d'outils | `false` |
| `--tool-call-parser` | Parseur d'appels d'outils (voir [Appel d'outils](../guides/tool-calling.md)) | None |

### Options de raisonnement

| Option | Description | Défaut |
|--------|-------------|---------|
| `--reasoning-parser` | Parseur pour les modèles de raisonnement (`qwen3`, `deepseek_r1`) | None |

### Options d'embeddings

| Option | Description | Défaut |
|--------|-------------|---------|
| `--embedding-model` | Précharge un modèle d'embedding au démarrage | None |

### Options MCP

| Option | Description | Défaut |
|--------|-------------|---------|
| `--mcp-config` | Chemin vers le fichier de configuration MCP | None |

## Configuration MCP

Créez le fichier `mcp.json` :

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-name", "arg1"],
      "env": {
        "ENV_VAR": "value"
      }
    }
  }
}
```

### Options du serveur MCP

| Champ | Description | Obligatoire |
|-------|-------------|-------------|
| `command` | Commande exécutable | Oui |
| `args` | Arguments de la commande | Oui |
| `env` | Variables d'environnement | Non |

## Options des requêtes API

### Complétions de chat

| Paramètre | Description | Défaut |
|-----------|-------------|---------|
| `model` | Nom du modèle | Obligatoire |
| `messages` | Messages du chat | Obligatoire |
| `max_tokens` | Nombre maximum de tokens à générer | 256 |
| `temperature` | Température d'échantillonnage | Défaut du modèle |
| `top_p` | Échantillonnage par noyau | Défaut du modèle |
| `stream` | Active le streaming | `true` |
| `stop` | Séquences d'arrêt | None |
| `tools` | Définitions des outils | None |
| `response_format` | Format de sortie (`json_object`, `json_schema`) | None |

### Options multimodales

| Paramètre | Description | Défaut |
|-----------|-------------|---------|
| `video_fps` | Images par seconde | 2.0 |
| `video_max_frames` | Nombre maximum d'images | 32 |

## Variables d'environnement

| Variable | Description |
|----------|-------------|
| `VLLM_MLX_TEST_MODEL` | Modèle par défaut pour les tests |
| `HF_TOKEN` | Token d'authentification HuggingFace |
| `OPENAI_API_KEY` | À définir avec n'importe quelle valeur pour la compatibilité SDK |

## Exemples de configurations

### Développement (utilisateur unique)

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Production (utilisateurs multiples)

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --api-key your-secret-key \
  --rate-limit 60 \
  --port 8000
```

### Avec appel d'outils

```bash
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral \
  --continuous-batching
```

### Avec les outils MCP

```bash
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --mcp-config mcp.json \
  --enable-auto-tool-choice \
  --tool-call-parser qwen \
  --continuous-batching
```

### Modèle de raisonnement

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit \
  --reasoning-parser qwen3 \
  --continuous-batching
```

### Avec embeddings

```bash
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --embedding-model mlx-community/multilingual-e5-small-mlx \
  --continuous-batching
```

### Débit élevé

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --stream-interval 5 \
  --max-num-seqs 256
```
