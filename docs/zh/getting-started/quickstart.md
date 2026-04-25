# 快速开始

## 选项 1：兼容 OpenAI 的服务器

启动服务器：

```bash
# Simple mode - maximum throughput for single user
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# Continuous batching - for multiple concurrent users
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

使用 OpenAI Python SDK：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

或使用 curl：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## 选项 2：直接使用 Python API

```python
from vllm_mlx.models import MLXLanguageModel

model = MLXLanguageModel("mlx-community/Llama-3.2-3B-Instruct-4bit")
model.load()

# Generate text
output = model.generate("What is the capital of France?", max_tokens=100)
print(output.text)

# Streaming
for chunk in model.stream_generate("Tell me a story"):
    print(chunk.text, end="", flush=True)
```

## 选项 3：Gradio 聊天界面

```bash
vllm-mlx-chat --served-model-name mlx-community/Llama-3.2-3B-Instruct-4bit
```

在 http://localhost:7860 打开网页界面。

## 多模态模型

如需图像或视频理解，请使用 VLM 模型：

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
    }],
    max_tokens=256
)
```

## 推理模型

将模型的思考过程与最终答案分离：

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)
print(response.choices[0].message.content)  # Final answer
```

## Embeddings

为语义搜索和 RAG 生成文本 embeddings：

```bash
vllm-mlx serve mlx-community/Qwen3-4B-4bit --embedding-model mlx-community/multilingual-e5-small-mlx
```

```python
response = client.embeddings.create(
    model="mlx-community/multilingual-e5-small-mlx",
    input="Hello world"
)
```

## 工具调用

为任意支持的模型启用函数调用：

```bash
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral
```

## 下一步

- [服务器指南](../guides/server.md) - 完整的服务器配置
- [Python API](../guides/python-api.md) - 直接使用 API
- [多模态指南](../guides/multimodal.md) - 图像与视频
- [音频指南](../guides/audio.md) - STT 与 TTS
- [Embeddings 指南](../guides/embeddings.md) - 文本 embeddings
- [推理模型](../guides/reasoning.md) - 思考模型
- [工具调用](../guides/tool-calling.md) - 函数调用
- [支持的模型](../reference/models.md) - 可用模型
