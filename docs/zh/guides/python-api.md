# Python API

通过 Python API 直接以编程方式访问 vllm-mlx。

## Language Models

### 基本用法

```python
from vllm_mlx.models import MLXLanguageModel

# Load model
model = MLXLanguageModel("mlx-community/Llama-3.2-3B-Instruct-4bit")
model.load()

# Generate text
output = model.generate("What is the capital of France?", max_tokens=100)
print(output.text)
```

### Streaming Generation

```python
for chunk in model.stream_generate("Tell me a story about a robot"):
    print(chunk.text, end="", flush=True)
```

### Chat 接口

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, who are you?"}
]
response = model.chat(messages)
print(response.text)
```

### 生成参数

```python
output = model.generate(
    prompt="Write a poem",
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    stop=["END", "\n\n"]
)
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `max_tokens` | 最大生成 token 数量 | 256 |
| `temperature` | 采样温度（0-2） | 0.7 |
| `top_p` | Nucleus sampling | 0.9 |
| `stop` | 停止序列 | None |

## Vision-Language Models

### 基本用法

```python
from vllm_mlx.models import MLXMultimodalLM

# Load model
mllm = MLXMultimodalLM("mlx-community/Qwen3-VL-4B-Instruct-3bit")
mllm.load()

# Describe an image
description = mllm.describe_image("photo.jpg")
print(description)
```

### 问答

```python
answer = mllm.answer_about_image("photo.jpg", "What color is the car?")
print(answer)
```

### 多图片输入

```python
output = mllm.generate(
    prompt="Compare these two images",
    images=["image1.jpg", "image2.jpg"]
)
print(output.text)
```

### 视频理解

```python
# From local file
output = mllm.generate(
    prompt="What is happening in this video?",
    videos=["video.mp4"],
    video_fps=2.0,
    video_max_frames=16
)
print(output.text)

# From URL
output = mllm.generate(
    prompt="Describe this video",
    videos=["https://example.com/video.mp4"],
    video_fps=2.0
)

# Convenience method
description = mllm.describe_video("video.mp4", fps=2.0)
```

### 视频参数

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `video_fps` | 每秒提取帧数 | 2.0 |
| `video_max_frames` | 最大处理帧数 | 32 |

## Engine API

针对高级使用场景，可直接使用 engine：

### Simple Engine

```python
from vllm_mlx.engine import SimpleEngine

engine = SimpleEngine("mlx-community/Llama-3.2-3B-Instruct-4bit")
await engine.start()

output = await engine.generate(
    prompt="Hello world",
    max_tokens=100
)
print(output.text)

await engine.stop()
```

### Batched Engine

```python
from vllm_mlx.engine import BatchedEngine

engine = BatchedEngine("mlx-community/Llama-3.2-3B-Instruct-4bit")
await engine.start()

# Multiple concurrent requests
output = await engine.generate(
    prompt="Hello world",
    max_tokens=100
)

await engine.stop()
```

## 输出格式

所有生成方法均返回 `GenerationOutput`：

```python
output = model.generate("Hello")

print(output.text)              # Generated text
print(output.prompt_tokens)     # Input token count
print(output.completion_tokens) # Output token count
print(output.finish_reason)     # "stop" or "length"
```

## 错误处理

```python
from vllm_mlx.models import MLXLanguageModel

try:
    model = MLXLanguageModel("invalid-model")
    model.load()
except Exception as e:
    print(f"Failed to load model: {e}")
```
