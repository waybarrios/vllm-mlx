# vLLM-MLX

**vLLM-like inference for Apple Silicon** - GPU-accelerated Text, Image, Video & Audio on Mac

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)](https://support.apple.com/en-us/HT211814)
[![GitHub](https://img.shields.io/badge/GitHub-waybarrios%2Fvllm--mlx-blue?logo=github)](https://github.com/waybarrios/vllm-mlx)

## Overview

vllm-mlx brings native Apple Silicon GPU acceleration to vLLM by integrating:

- **[MLX](https://github.com/ml-explore/mlx)**: Apple's ML framework with unified memory and Metal kernels
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)**: Optimized LLM inference with KV cache and quantization
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)**: Vision-language models for multimodal inference
- **[mlx-audio](https://github.com/Blaizzy/mlx-audio)**: Speech-to-Text and Text-to-Speech with native voices

## Features

- **Multimodal** - Text, Image, Video & Audio in one platform
- **Native GPU acceleration** on Apple Silicon (M1, M2, M3, M4)
- **Native TTS voices** - Spanish, French, Chinese, Japanese + 5 more languages
- **OpenAI API compatible** - drop-in replacement for OpenAI client
- **Reasoning Models** - extract thinking process from Qwen3, DeepSeek-R1
- **MCP Tool Calling** - integrate external tools via Model Context Protocol
- **Paged KV Cache** - memory-efficient caching with prefix sharing
- **Continuous Batching** - high throughput for multiple concurrent users

## Quick Start

### Installation

```bash
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx
pip install -e .
```

### Start Server

```bash
# Simple mode (single user, max throughput)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# Continuous batching (multiple users)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching

# With API key authentication
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --api-key your-secret-key
```

### Use with OpenAI SDK

```python
from openai import OpenAI

# Without API key (local development)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# With API key (production)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-secret-key")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Multimodal (Images & Video)

```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }]
)
```

### Audio (TTS/STT)

```bash
# Install audio dependencies
pip install vllm-mlx[audio]
python -m spacy download en_core_web_sm
brew install espeak-ng  # macOS, for non-English languages
```

```bash
# Text-to-Speech (English)
python examples/tts_example.py "Hello, how are you?" --play

# Text-to-Speech (Spanish)
python examples/tts_multilingual.py "Hola mundo" --lang es --play

# List available models and languages
python examples/tts_multilingual.py --list-models
python examples/tts_multilingual.py --list-languages
```

**Supported TTS Models:**
| Model | Languages | Description |
|-------|-----------|-------------|
| Kokoro | EN, ES, FR, JA, ZH, IT, PT, HI | Fast, 82M params, 11 voices |
| Chatterbox | 15+ languages | Expressive, voice cloning |
| VibeVoice | EN | Realtime, low latency |
| VoxCPM | ZH, EN | High quality Chinese/English |

### Reasoning Models

Extract the thinking process from reasoning models like Qwen3 and DeepSeek-R1:

```bash
# Start server with reasoning parser
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)

# Access reasoning separately from the answer
print("Thinking:", response.choices[0].message.reasoning)
print("Answer:", response.choices[0].message.content)
```

**Supported Parsers:**
| Parser | Models | Description |
|--------|--------|-------------|
| `qwen3` | Qwen3 series | Requires both `<think>` and `</think>` tags |
| `deepseek_r1` | DeepSeek-R1 | Handles implicit `<think>` tag |

## Documentation

For full documentation, see the [docs](docs/) directory:

- **Getting Started**
  - [Installation](docs/getting-started/installation.md)
  - [Quick Start](docs/getting-started/quickstart.md)

- **User Guides**
  - [OpenAI-Compatible Server](docs/guides/server.md)
  - [Python API](docs/guides/python-api.md)
  - [Multimodal (Images & Video)](docs/guides/multimodal.md)
  - [Audio (STT/TTS)](docs/guides/audio.md)
  - [Reasoning Models](docs/guides/reasoning.md)
  - [MCP & Tool Calling](docs/guides/mcp-tools.md)
  - [Continuous Batching](docs/guides/continuous-batching.md)

- **Reference**
  - [CLI Commands](docs/reference/cli.md)
  - [Supported Models](docs/reference/models.md)
  - [Configuration](docs/reference/configuration.md)

- **Benchmarks**
  - [LLM Benchmarks](docs/benchmarks/llm.md)
  - [Image Benchmarks](docs/benchmarks/image.md)
  - [Video Benchmarks](docs/benchmarks/video.md)
  - [Audio Benchmarks](docs/benchmarks/audio.md)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    vLLM API Layer                       │
│           (OpenAI-compatible interface)                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    MLXPlatform                          │
│       (vLLM platform plugin for Apple Silicon)          │
└─────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│     mlx-lm       │ │   mlx-vlm    │ │    mlx-audio     │
│  (LLM inference) │ │ (Vision+LLM) │ │   (TTS + STT)    │
└──────────────────┘ └──────────────┘ └──────────────────┘
          │                │                  │
          └────────────────┴──────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                        MLX                              │
│         (Apple ML Framework - Metal kernels)            │
└─────────────────────────────────────────────────────────┘
```

## Performance

**LLM Performance (M4 Max, 128GB):**

| Model | Speed | Memory |
|-------|-------|--------|
| Qwen3-0.6B-8bit | 402 tok/s | 0.7 GB |
| Llama-3.2-1B-4bit | 464 tok/s | 0.7 GB |
| Llama-3.2-3B-4bit | 200 tok/s | 1.8 GB |

**Continuous Batching (5 concurrent requests):**

| Model | Single | Batched | Speedup |
|-------|--------|---------|---------|
| Qwen3-0.6B-8bit | 328 tok/s | 1112 tok/s | **3.4x** |
| Llama-3.2-1B-4bit | 299 tok/s | 613 tok/s | **2.0x** |

**Audio - Speech-to-Text (M4 Max, 128GB):**

| Model | RTF* | Use Case |
|-------|------|----------|
| whisper-tiny | **197x** | Real-time, low latency |
| whisper-large-v3-turbo | **55x** | Best quality/speed balance |
| whisper-large-v3 | **24x** | Highest accuracy |

*RTF = Real-Time Factor. RTF of 100x means 1 minute transcribes in ~0.6 seconds.

See [benchmarks](docs/benchmarks/) for detailed results.

## Gemma 3 Support

vllm-mlx includes native support for Gemma 3 vision models. Gemma 3 is automatically detected as MLLM.

### Usage

```bash
# Start server with Gemma 3
vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000

# Verify it loaded as MLLM (not LLM)
curl http://localhost:8000/health
# Should show: "model_type": "mllm"
```

### Long Context Patch (mlx-vlm)

Gemma 3's default `sliding_window=1024` limits context to ~10K tokens on Apple Silicon (Metal GPU timeout at higher context). To enable longer context (up to ~50K tokens), patch mlx-vlm:

**Location:** `~/.../site-packages/mlx_vlm/models/gemma3/language.py`

Find the `make_cache` method and replace with:

```python
def make_cache(self):
    import os
    # Set GEMMA3_SLIDING_WINDOW=8192 for ~40K context
    # Set GEMMA3_SLIDING_WINDOW=0 for ~50K context (full KVCache)
    sliding_window = int(os.environ.get('GEMMA3_SLIDING_WINDOW', self.config.sliding_window))

    caches = []
    for i in range(self.config.num_hidden_layers):
        if (
            i % self.config.sliding_window_pattern
            == self.config.sliding_window_pattern - 1
        ):
            caches.append(KVCache())
        elif sliding_window == 0:
            caches.append(KVCache())  # Full context for all layers
        else:
            caches.append(RotatingKVCache(max_size=sliding_window, keep=0))
    return caches
```

**Usage:**

```bash
# Default (~10K max context)
vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000

# Extended context (~40K max)
GEMMA3_SLIDING_WINDOW=8192 vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000

# Maximum context (~50K max)
GEMMA3_SLIDING_WINDOW=0 vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000
```

**Benchmark Results (M4 Max 128GB):**

| Setting | Max Context | Memory |
|---------|-------------|--------|
| Default (1024) | ~10K tokens | ~16GB |
| `GEMMA3_SLIDING_WINDOW=8192` | ~40K tokens | ~25GB |
| `GEMMA3_SLIDING_WINDOW=0` | ~50K tokens | ~35GB |

## Contributing

We welcome contributions! See [Contributing Guide](docs/development/contributing.md) for details.

- Bug fixes and improvements
- Performance optimizations
- Documentation improvements
- Benchmarks on different Apple Silicon chips

Submit PRs to: [https://github.com/waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx)

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Citation

If you use vLLM-MLX in your research or project, please cite:

```bibtex
@software{vllm_mlx2025,
  author = {Barrios, Wayner},
  title = {vLLM-MLX: Apple Silicon MLX Backend for vLLM},
  year = {2025},
  url = {https://github.com/waybarrios/vllm-mlx},
  note = {Native GPU-accelerated LLM and vision-language model inference on Apple Silicon}
}
```

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - LLM inference library
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision-language models
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - Text-to-Speech and Speech-to-Text
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
