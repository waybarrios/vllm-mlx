# 支持的模型

所有来自 [mlx-community on HuggingFace](https://huggingface.co/mlx-community/models) 的量化模型均兼容。

在以下地址浏览数千个预优化模型：**https://huggingface.co/mlx-community/models**

## 语言模型（通过 mlx-lm）

| 模型系列 | 规格 | 量化方式 |
|--------------|-------|--------------|
| Llama 3.x, 4.x | 1B, 3B, 8B, 70B | 4-bit |
| Mistral / Devstral | 7B, Mixtral 8x7B | 4-bit, 8-bit |
| Qwen2/Qwen3 | 0.5B 至 72B | 多种 |
| DeepSeek V3, R1 | 7B, 33B, 67B | 4-bit |
| Gemma 2, 3, 4 | 2B, 9B, 27B | 4-bit |
| GLM-4.7 | Flash, Base | 4-bit, 8-bit |
| Kimi K2 | 多种 | 4-bit |
| Phi-3 | 3.8B, 14B | 4-bit |
| Granite 3.x, 4.x | 多种 | 4-bit |
| Nemotron | 3 Nano 30B | 6-bit |

### 推荐模型

| 使用场景 | 模型 | 内存 |
|----------|-------|--------|
| 快速/轻量 | `mlx-community/Qwen3-0.6B-8bit` | ~0.7 GB |
| 均衡 | `mlx-community/Llama-3.2-3B-Instruct-4bit` | ~1.8 GB |
| 高质量 | `mlx-community/Llama-3.1-8B-Instruct-4bit` | ~4.5 GB |
| 大型 | `mlx-community/Qwen3-30B-A3B-4bit` | ~16 GB |

## 多模态模型（通过 mlx-vlm）

| 模型系列 | 示例模型 |
|--------------|----------------|
| **Qwen-VL** | `Qwen3-VL-4B-Instruct-3bit`, `Qwen3-VL-8B-Instruct-4bit`, `Qwen2-VL-2B/7B-Instruct-4bit` |
| **LLaVA** | `llava-1.5-7b-4bit`, `llava-v1.6-mistral-7b-4bit`, `llava-llama-3-8b-v1_1-4bit` |
| **Idefics** | `Idefics3-8B-Llama3-4bit`, `idefics2-8b-4bit` |
| **Gemma 4** | `gemma-4-e2b-it-mxfp4`（视觉 + 音频） |
| **PaliGemma** | `paligemma2-3b-mix-224-4bit`, `paligemma-3b-mix-224-8bit` |
| **Pixtral** | `pixtral-12b-4bit`, `pixtral-12b-8bit` |
| **Molmo** | `Molmo-7B-D-0924-4bit`, `Molmo-7B-D-0924-8bit` |
| **Phi-3 Vision** | `Phi-3-vision-128k-instruct-4bit` |
| **DeepSeek-VL** | `deepseek-vl-7b-chat-4bit`, `deepseek-vl2-small-4bit` |

### 推荐 VLM 模型

| 使用场景 | 模型 | 内存 |
|----------|-------|--------|
| 快速/轻量 | `mlx-community/Qwen3-VL-4B-Instruct-3bit` | ~3 GB |
| 均衡 | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | ~6 GB |
| 高质量 | `mlx-community/Qwen3-VL-30B-A3B-Instruct-6bit` | ~20 GB |

## Embedding 模型（通过 mlx-embeddings）

| 模型系列 | 示例模型 |
|--------------|----------------|
| **BERT** | `mlx-community/bert-base-uncased-mlx` |
| **XLM-RoBERTa** | `mlx-community/multilingual-e5-small-mlx`, `mlx-community/multilingual-e5-large-mlx` |
| **ModernBERT** | `mlx-community/ModernBERT-base-mlx` |

## 音频模型（通过 mlx-audio）

| 类型 | 模型系列 | 示例模型 |
|------|--------------|----------------|
| **STT** | Whisper | `mlx-community/whisper-large-v3-turbo` |
| **STT** | Parakeet | `mlx-community/parakeet-tdt-0.6b-v2` |
| **TTS** | Kokoro | `prince-canuma/Kokoro-82M` |
| **TTS** | Chatterbox | `chatterbox/chatterbox-tts-0.1` |

## 模型自动检测

vllm-mlx 通过名称模式自动检测多模态模型：
- 包含 "VL"、"Vision"、"vision"
- 包含 "llava"、"idefics"、"paligemma"
- 包含 "pixtral"、"molmo"、"deepseek-vl"
- 包含 "MedGemma"、"Gemma-3"、"Gemma-4"（多模态变体）

## 使用模型

### 从 HuggingFace 加载

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit
```

### 本地路径

```bash
vllm-mlx serve /path/to/local/model
```

## 查找模型

按以下关键词筛选 mlx-community 模型：
- **LLM**：`Llama`、`Qwen`、`Mistral`、`Phi`、`Gemma`、`DeepSeek`、`GLM`、`Kimi`、`Granite`、`Nemotron`
- **VLM**：`-VL-`、`llava`、`paligemma`、`pixtral`、`molmo`、`idefics`、`deepseek-vl`、`MedGemma`
- **Embedding**：`e5`、`bert`、`ModernBERT`
- **规格**：`1B`、`3B`、`7B`、`8B`、`70B`
- **量化方式**：`4bit`、`8bit`、`bf16`
