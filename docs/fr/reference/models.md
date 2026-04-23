# Modèles pris en charge

Tous les modèles quantifiés de [mlx-community sur HuggingFace](https://huggingface.co/mlx-community/models) sont compatibles.

Parcourez des milliers de modèles pré-optimisés à l'adresse : **https://huggingface.co/mlx-community/models**

## Modèles de langage (via mlx-lm)

| Famille de modèles | Tailles | Quantification |
|--------------------|---------|----------------|
| Llama 3.x, 4.x | 1B, 3B, 8B, 70B | 4-bit |
| Mistral / Devstral | 7B, Mixtral 8x7B | 4-bit, 8-bit |
| Qwen2/Qwen3 | 0.5B à 72B | Variable |
| DeepSeek V3, R1 | 7B, 33B, 67B | 4-bit |
| Gemma 2, 3, 4 | 2B, 9B, 27B | 4-bit |
| GLM-4.7 | Flash, Base | 4-bit, 8-bit |
| Kimi K2 | Variable | 4-bit |
| Phi-3 | 3.8B, 14B | 4-bit |
| Granite 3.x, 4.x | Variable | 4-bit |
| Nemotron | 3 Nano 30B | 6-bit |

### Modèles recommandés

| Cas d'utilisation | Modèle | Mémoire |
|-------------------|--------|---------|
| Rapide / léger | `mlx-community/Qwen3-0.6B-8bit` | ~0,7 Go |
| Équilibré | `mlx-community/Llama-3.2-3B-Instruct-4bit` | ~1,8 Go |
| Qualité | `mlx-community/Llama-3.1-8B-Instruct-4bit` | ~4,5 Go |
| Grand modèle | `mlx-community/Qwen3-30B-A3B-4bit` | ~16 Go |

## Modèles multimodaux (via mlx-vlm)

| Famille de modèles | Exemples de modèles |
|--------------------|---------------------|
| **Qwen-VL** | `Qwen3-VL-4B-Instruct-3bit`, `Qwen3-VL-8B-Instruct-4bit`, `Qwen2-VL-2B/7B-Instruct-4bit` |
| **LLaVA** | `llava-1.5-7b-4bit`, `llava-v1.6-mistral-7b-4bit`, `llava-llama-3-8b-v1_1-4bit` |
| **Idefics** | `Idefics3-8B-Llama3-4bit`, `idefics2-8b-4bit` |
| **Gemma 4** | `gemma-4-e2b-it-mxfp4` (vision + audio) |
| **PaliGemma** | `paligemma2-3b-mix-224-4bit`, `paligemma-3b-mix-224-8bit` |
| **Pixtral** | `pixtral-12b-4bit`, `pixtral-12b-8bit` |
| **Molmo** | `Molmo-7B-D-0924-4bit`, `Molmo-7B-D-0924-8bit` |
| **Phi-3 Vision** | `Phi-3-vision-128k-instruct-4bit` |
| **DeepSeek-VL** | `deepseek-vl-7b-chat-4bit`, `deepseek-vl2-small-4bit` |

### Modèles VLM recommandés

| Cas d'utilisation | Modèle | Mémoire |
|-------------------|--------|---------|
| Rapide / léger | `mlx-community/Qwen3-VL-4B-Instruct-3bit` | ~3 Go |
| Équilibré | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | ~6 Go |
| Qualité | `mlx-community/Qwen3-VL-30B-A3B-Instruct-6bit` | ~20 Go |

## Modèles d'embeddings (via mlx-embeddings)

| Famille de modèles | Exemples de modèles |
|--------------------|---------------------|
| **BERT** | `mlx-community/bert-base-uncased-mlx` |
| **XLM-RoBERTa** | `mlx-community/multilingual-e5-small-mlx`, `mlx-community/multilingual-e5-large-mlx` |
| **ModernBERT** | `mlx-community/ModernBERT-base-mlx` |

## Modèles audio (via mlx-audio)

| Type | Famille de modèles | Exemples de modèles |
|------|--------------------|---------------------|
| **STT** | Whisper | `mlx-community/whisper-large-v3-turbo` |
| **STT** | Parakeet | `mlx-community/parakeet-tdt-0.6b-v2` |
| **TTS** | Kokoro | `prince-canuma/Kokoro-82M` |
| **TTS** | Chatterbox | `chatterbox/chatterbox-tts-0.1` |

## Détection automatique des modèles

vllm-mlx détecte automatiquement les modèles multimodaux selon des motifs dans leur nom :
- Contient "VL", "Vision", "vision"
- Contient "llava", "idefics", "paligemma"
- Contient "pixtral", "molmo", "deepseek-vl"
- Contient "MedGemma", "Gemma-3", "Gemma-4" (variantes multimodales)

## Utilisation des modèles

### Depuis HuggingFace

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Chemin local

```bash
vllm-mlx serve /path/to/local/model
```

## Recherche de modèles

Filtrez les modèles mlx-community par :
- **LLM** : `Llama`, `Qwen`, `Mistral`, `Phi`, `Gemma`, `DeepSeek`, `GLM`, `Kimi`, `Granite`, `Nemotron`
- **VLM** : `-VL-`, `llava`, `paligemma`, `pixtral`, `molmo`, `idefics`, `deepseek-vl`, `MedGemma`
- **Embedding** : `e5`, `bert`, `ModernBERT`
- **Taille** : `1B`, `3B`, `7B`, `8B`, `70B`
- **Quantification** : `4bit`, `8bit`, `bf16`
