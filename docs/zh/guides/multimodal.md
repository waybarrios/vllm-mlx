# 多模态模型（图像与视频）

vllm-mlx 支持用于图像和视频理解的视觉语言模型。

## 支持的模型

- Qwen3-VL（推荐）
- Qwen2-VL
- Gemma 3
- LLaVA
- Idefics
- PaliGemma
- Pixtral
- Molmo
- DeepSeek-VL

## 启动多模态服务器

```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

名称中包含 "VL"、"Vision" 或 "mllm" 的模型会被自动识别为多模态模型。

## 图像分析

### 通过 OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Image from URL
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }],
    max_tokens=256
)
print(response.choices[0].message.content)
```

### Base64 图像

```python
import base64

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

base64_image = encode_image("photo.jpg")
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }]
)
```

### 通过 curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }],
    "max_tokens": 256
  }'
```

## 视频分析

### 通过 OpenAI SDK

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }],
    max_tokens=512
)
```

### 视频参数

通过额外的请求体参数控制帧提取：

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this video"},
            {"type": "video_url", "video_url": {"url": "video.mp4"}}
        ]
    }],
    extra_body={
        "video_fps": 2.0,
        "video_max_frames": 32
    }
)
```

### 通过 curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this video"},
        {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
      ]
    }],
    "video_fps": 2.0,
    "video_max_frames": 16
  }'
```

## 支持的格式

### 图像

| 格式 | 示例 |
|------|------|
| URL | `{"type": "image_url", "image_url": {"url": "https://..."}}` |
| 本地文件 | `{"type": "image_url", "image_url": {"url": "/path/to/image.jpg"}}` |
| Base64 | `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}` |

### 视频

| 格式 | 示例 |
|------|------|
| URL | `{"type": "video_url", "video_url": {"url": "https://..."}}` |
| 本地文件 | `{"type": "video", "video": "/path/to/video.mp4"}` |
| Base64 | `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}` |

## Python API

```python
from vllm_mlx.models import MLXMultimodalLM

mllm = MLXMultimodalLM("mlx-community/Qwen3-VL-4B-Instruct-3bit")
mllm.load()

# Image
description = mllm.describe_image("photo.jpg")

# Video
description = mllm.describe_video("video.mp4", fps=2.0)

# Custom prompt
output = mllm.generate(
    prompt="Compare these images",
    images=["img1.jpg", "img2.jpg"]
)
```

## 性能建议

### 图像
- 分辨率越小，处理速度越快（224x224 对比 1920x1080）
- 根据任务选择合适的分辨率

### 视频
- 帧率越低，处理速度越快
- 帧数越少，内存占用越低
- 64 帧是实际可用的最大值（96 帧及以上会导致 GPU 超时）

## 基准测试

在配备 128 GB 统一内存的 Apple M4 Max 上测试。

### Qwen3-VL-4B-Instruct-3bit

| 分辨率 | 耗时 | token 数 | 速度 | 内存 |
|--------|------|----------|------|------|
| 224x224 | 0.87s | 124 | 143 tok/s | 2.6 GB |
| 448x448 | 1.01s | 107 | 106 tok/s | 3.1 GB |
| 768x768 | 1.42s | 127 | 89 tok/s | 3.4 GB |
| 1024x1024 | 1.85s | 116 | 63 tok/s | 3.6 GB |

### Qwen3-VL-8B-Instruct-4bit

| 分辨率 | 耗时 | token 数 | 速度 | 内存 |
|--------|------|----------|------|------|
| 224x224 | 1.08s | 78 | 73 tok/s | 5.6 GB |
| 448x448 | 1.41s | 70 | 50 tok/s | 6.1 GB |
| 768x768 | 2.06s | 91 | 44 tok/s | 6.5 GB |
| 1024x1024 | 3.02s | 76 | 25 tok/s | 7.6 GB |

### Gemma 3 4B 4bit

| 分辨率 | 耗时 | token 数 | 速度 | 内存 |
|--------|------|----------|------|------|
| 224x224 | 0.95s | 30 | 32 tok/s | 5.2 GB |
| 448x448 | 0.99s | 34 | 34 tok/s | 5.2 GB |
| 768x768 | 0.99s | 32 | 32 tok/s | 5.2 GB |
| 1024x1024 | 0.95s | 28 | 29 tok/s | 5.2 GB |

### 运行基准测试

```bash
# Quick benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --quick

# Full benchmark with more resolutions
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit

# Video benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --video
```

## MLLM Cache

vllm-mlx 为多模态模型内置了 prefix cache 系统，可以显著加速对相同图像的重复请求。

### 工作原理

向模型发送图像时，视觉编码器会将其处理为嵌入向量，该过程通常需要 1 到 2 秒。MLLM Cache 会同时存储这些嵌入向量和 KV cache 状态，因此后续使用相同图像的请求可以完全跳过视觉编码器。

该缓存采用基于内容的哈希（类似 LMCache）来识别相同图像，无论图像以何种方式提供（URL、base64 还是文件路径）。

### 启用缓存

```bash
# Enable with default settings (512 MB max)
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --enable-mllm-cache

# With custom memory limit
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit \
    --enable-mllm-cache \
    --mllm-cache-max-mb 1024
```

### Python API

```python
from vllm_mlx.mllm_cache import MLLMPrefixCacheManager

# Create cache manager
cache = MLLMPrefixCacheManager(max_memory_mb=512)

# Store embeddings and KV cache after processing
cache.store(
    images=["photo.jpg"],
    prompt="Describe this image",
    vision_embeddings=embeddings,
    kv_cache=kv_state,
    num_tokens=128
)

# Fetch from cache on subsequent requests
entry, match_len = cache.fetch(images=["photo.jpg"], prompt="Describe this image")
if entry:
    # Use cached embeddings, skip vision encoder
    embeddings = entry.vision_embeddings
    kv_state = entry.kv_cache
```

### 缓存统计

```python
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Memory used: {stats.memory_used_mb:.1f} MB")
print(f"Tokens saved: {stats.tokens_saved}")
```

### 内存管理

当达到内存上限时，缓存采用 LRU（最近最少使用）策略进行淘汰。每个缓存条目记录以下信息：

- 视觉嵌入向量大小
- 每层 KV cache 大小
- 用于 LRU 排序的访问频率

当内存压力出现时，最近最少访问的条目会被优先淘汰。

## Gradio 聊天界面

如需交互式多模态对话：

```bash
vllm-mlx-chat --served-model-name mlx-community/Qwen3-VL-4B-Instruct-3bit
```

支持拖放图像和视频。
