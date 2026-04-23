# Modelos compatibles

Todos los modelos cuantizados de [mlx-community en HuggingFace](https://huggingface.co/mlx-community/models) son compatibles.

Explora miles de modelos preoptimizados en: **https://huggingface.co/mlx-community/models**

## Modelos de lenguaje (vía mlx-lm)

| Familia de modelos | Tamaños | Cuantización |
|--------------------|---------|--------------|
| Llama 3.x, 4.x | 1B, 3B, 8B, 70B | 4-bit |
| Mistral / Devstral | 7B, Mixtral 8x7B | 4-bit, 8-bit |
| Qwen2/Qwen3 | 0.5B a 72B | Varios |
| DeepSeek V3, R1 | 7B, 33B, 67B | 4-bit |
| Gemma 2, 3, 4 | 2B, 9B, 27B | 4-bit |
| GLM-4.7 | Flash, Base | 4-bit, 8-bit |
| Kimi K2 | Varios | 4-bit |
| Phi-3 | 3.8B, 14B | 4-bit |
| Granite 3.x, 4.x | Varios | 4-bit |
| Nemotron | 3 Nano 30B | 6-bit |

### Modelos recomendados

| Caso de uso | Modelo | Memoria |
|-------------|--------|---------|
| Rápido / Liviano | `mlx-community/Qwen3-0.6B-8bit` | ~0.7 GB |
| Equilibrado | `mlx-community/Llama-3.2-3B-Instruct-4bit` | ~1.8 GB |
| Calidad | `mlx-community/Llama-3.1-8B-Instruct-4bit` | ~4.5 GB |
| Grande | `mlx-community/Qwen3-30B-A3B-4bit` | ~16 GB |

## Modelos multimodales (vía mlx-vlm)

| Familia de modelos | Modelos de ejemplo |
|--------------------|--------------------|
| **Qwen-VL** | `Qwen3-VL-4B-Instruct-3bit`, `Qwen3-VL-8B-Instruct-4bit`, `Qwen2-VL-2B/7B-Instruct-4bit` |
| **LLaVA** | `llava-1.5-7b-4bit`, `llava-v1.6-mistral-7b-4bit`, `llava-llama-3-8b-v1_1-4bit` |
| **Idefics** | `Idefics3-8B-Llama3-4bit`, `idefics2-8b-4bit` |
| **Gemma 4** | `gemma-4-e2b-it-mxfp4` (visión + audio) |
| **PaliGemma** | `paligemma2-3b-mix-224-4bit`, `paligemma-3b-mix-224-8bit` |
| **Pixtral** | `pixtral-12b-4bit`, `pixtral-12b-8bit` |
| **Molmo** | `Molmo-7B-D-0924-4bit`, `Molmo-7B-D-0924-8bit` |
| **Phi-3 Vision** | `Phi-3-vision-128k-instruct-4bit` |
| **DeepSeek-VL** | `deepseek-vl-7b-chat-4bit`, `deepseek-vl2-small-4bit` |

### Modelos VLM recomendados

| Caso de uso | Modelo | Memoria |
|-------------|--------|---------|
| Rápido / Liviano | `mlx-community/Qwen3-VL-4B-Instruct-3bit` | ~3 GB |
| Equilibrado | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | ~6 GB |
| Calidad | `mlx-community/Qwen3-VL-30B-A3B-Instruct-6bit` | ~20 GB |

## Modelos de embeddings (vía mlx-embeddings)

| Familia de modelos | Modelos de ejemplo |
|--------------------|--------------------|
| **BERT** | `mlx-community/bert-base-uncased-mlx` |
| **XLM-RoBERTa** | `mlx-community/multilingual-e5-small-mlx`, `mlx-community/multilingual-e5-large-mlx` |
| **ModernBERT** | `mlx-community/ModernBERT-base-mlx` |

## Modelos de audio (vía mlx-audio)

| Tipo | Familia de modelos | Modelos de ejemplo |
|------|--------------------|--------------------|
| **STT** | Whisper | `mlx-community/whisper-large-v3-turbo` |
| **STT** | Parakeet | `mlx-community/parakeet-tdt-0.6b-v2` |
| **TTS** | Kokoro | `prince-canuma/Kokoro-82M` |
| **TTS** | Chatterbox | `chatterbox/chatterbox-tts-0.1` |

## Detección de modelos

vllm-mlx detecta automáticamente los modelos multimodales por patrones en el nombre:
- Contiene "VL", "Vision", "vision"
- Contiene "llava", "idefics", "paligemma"
- Contiene "pixtral", "molmo", "deepseek-vl"
- Contiene "MedGemma", "Gemma-3", "Gemma-4" (variantes multimodales)

## Usar modelos

### Desde HuggingFace

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Ruta local

```bash
vllm-mlx serve /path/to/local/model
```

## Buscar modelos

Filtra los modelos de mlx-community por:
- **LLM**: `Llama`, `Qwen`, `Mistral`, `Phi`, `Gemma`, `DeepSeek`, `GLM`, `Kimi`, `Granite`, `Nemotron`
- **VLM**: `-VL-`, `llava`, `paligemma`, `pixtral`, `molmo`, `idefics`, `deepseek-vl`, `MedGemma`
- **Embedding**: `e5`, `bert`, `ModernBERT`
- **Tamaño**: `1B`, `3B`, `7B`, `8B`, `70B`
- **Cuantización**: `4bit`, `8bit`, `bf16`
