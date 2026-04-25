# Embeddings

vllm-mlx 通过 [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) 支持文本 embeddings，提供与 OpenAI 兼容的 `/v1/embeddings` 接口。

## 安装

```bash
pip install mlx-embeddings>=0.0.5
```

## 快速入门

### 启动带有 embedding 模型的服务器

```bash
# 启动时预加载指定的 embedding 模型
vllm-mlx serve my-llm-model --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

如果不使用 `--embedding-model`，embedding 模型会在第一次请求时按需加载，但仅限于内置的请求时许可列表中的模型。

### 使用 OpenAI SDK 生成 embeddings

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 单条文本
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input="Hello world"
)
print(response.data[0].embedding[:5])  # First 5 dimensions

# 批量文本
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input=[
        "I love machine learning",
        "Deep learning is fascinating",
        "Natural language processing rocks"
    ]
)
for item in response.data:
    print(f"Text {item.index}: {len(item.embedding)} dimensions")
```

### 使用 curl

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/all-MiniLM-L6-v2-4bit",
    "input": ["Hello world", "How are you?"]
  }'
```

## 支持的模型

请求时支持的模型：

| 模型 | 适用场景 | 规模 |
|-------|----------|------|
| `mlx-community/all-MiniLM-L6-v2-4bit` | 快速、轻量 | 小 |
| `mlx-community/embeddinggemma-300m-6bit` | 高质量 | 300M |
| `mlx-community/bge-large-en-v1.5-4bit` | 英文效果最佳 | 大 |
| `mlx-community/multilingual-e5-small-mlx` | 多语言检索 | 小 |
| `mlx-community/multilingual-e5-large-mlx` | 多语言检索 | 大 |
| `mlx-community/bert-base-uncased-mlx` | 通用 BERT 基准 | 基础 |
| `mlx-community/ModernBERT-base-mlx` | ModernBERT 基准 | 基础 |

其他 embedding 模型需要在启动服务器时通过 `--embedding-model` 指定。

## 模型管理

### 按需加载

默认情况下，embedding 模型在第一次收到 `/v1/embeddings` 请求时加载。你可以在上述请求时支持的模型之间切换，切换后旧模型会自动卸载。

### 启动时预加载

使用 `--embedding-model` 可在启动时加载模型。设置该参数后，只有该指定模型可用于 embeddings：

```bash
vllm-mlx serve my-llm-model --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

请求其他模型将返回 400 错误。

## API 参考

### POST /v1/embeddings

为给定的输入文本生成 embeddings。

**请求体：**

| 字段 | 类型 | 是否必填 | 描述 |
|-------|------|----------|-------------|
| `model` | string | 是 | 支持的 embedding 模型 ID，或使用 `--embedding-model` 时启动时固定的模型 |
| `input` | string 或 list[string] | 是 | 待嵌入的文本 |

**响应：**

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.023, -0.982, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.112, -0.543, ...]}
  ],
  "model": "mlx-community/all-MiniLM-L6-v2-4bit",
  "usage": {"prompt_tokens": 12, "total_tokens": 12}
}
```

## Python API

### 不启动服务器直接使用

```python
from vllm_mlx.embedding import EmbeddingEngine

engine = EmbeddingEngine("mlx-community/all-MiniLM-L6-v2-4bit")
engine.load()

vectors = engine.embed(["Hello world", "How are you?"])
print(f"Dimensions: {len(vectors[0])}")

tokens = engine.count_tokens(["Hello world"])
print(f"Token count: {tokens}")
```

## 常见问题

### mlx-embeddings 未安装

```
pip install mlx-embeddings>=0.0.5
```

### 找不到模型

请确认模型名称与上方请求时支持的 ID 之一匹配，或在启动服务器时通过 `--embedding-model` 指定自定义模型。你也可以提前下载支持的模型：

```bash
huggingface-cli download mlx-community/all-MiniLM-L6-v2-4bit
```
