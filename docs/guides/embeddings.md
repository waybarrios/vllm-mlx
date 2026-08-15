# Embeddings

vllm-mlx supports text embeddings using [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings), providing an OpenAI-compatible `/v1/embeddings` endpoint.

## Installation

```bash
pip install mlx-embeddings>=0.0.5
```

## Quick Start

### Start the server with an embedding model

```bash
# Pre-load a specific embedding model at startup
vllm-mlx serve my-llm-model --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

If you don't use `--embedding-model`, the embedding model is loaded lazily on the first request, but only from the built-in request-time allowlist.

### Generate embeddings with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Single text
response = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input="Hello world"
)
print(response.data[0].embedding[:5])  # First 5 dimensions

# Batch of texts
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

### Using curl

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/all-MiniLM-L6-v2-4bit",
    "input": ["Hello world", "How are you?"]
  }'
```

## Supported Models

Supported request-time models:

| Model | Use Case | Size |
|-------|----------|------|
| `mlx-community/all-MiniLM-L6-v2-4bit` | Fast, compact | Small |
| `mlx-community/embeddinggemma-300m-6bit` | High quality | 300M |
| `mlx-community/bge-large-en-v1.5-4bit` | Best for English | Large |
| `mlx-community/multilingual-e5-small-mlx` | Multilingual retrieval | Small |
| `mlx-community/multilingual-e5-large-mlx` | Multilingual retrieval | Large |
| `mlx-community/bert-base-uncased-mlx` | General BERT baseline | Base |
| `mlx-community/ModernBERT-base-mlx` | ModernBERT baseline | Base |

Other embedding models require `--embedding-model` at server startup.

## Model Management

### Lazy loading

By default, the embedding model is loaded on the first `/v1/embeddings` request. You can switch between the supported request-time models above, and the previous model will be unloaded automatically.

### Pre-loading at startup

Use `--embedding-model` to load a model at startup. When this flag is set, only that specific model can be used for embeddings:

```bash
vllm-mlx serve my-llm-model --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

Requesting a different model will return a 400 error.

### Input length: ceiling and overflow policy

The truncation length used to tokenize input defaults to each model's own context window (`config.max_position_embeddings`, or a sanitized `tokenizer.model_max_length`, falling back to 512 when neither is usable). Classic 512-token BERT-family models are unaffected by this default.

Operators can additionally cap that value with `--embedding-max-length`, and choose what happens when an input still exceeds the effective limit with `--embedding-overflow-policy`:

```bash
vllm-mlx serve my-llm-model \
  --embedding-model mlx-community/embeddinggemma-300m-6bit \
  --embedding-max-length 1024 \
  --embedding-overflow-policy error
```

| Flag | Default | Description |
|------|---------|-------------|
| `--embedding-max-length` | `auto` | `auto` (or omitting the flag) uses the model-aware default above. A positive integer imposes a lower deployment-wide ceiling — the effective limit is `min(model-aware default, ceiling)`, so it can only lower the limit, never raise it above what the model supports. |
| `--embedding-overflow-policy` | `truncate` | `truncate` keeps today's behavior (input is truncated to the effective limit) but logs a warning and increments the `vllm_mlx_embedding_truncated_total` metric with the model, original token count, and effective limit. `error` rejects over-limit inputs instead, with a structured 400 response. |

**Memory implications of large context windows.** `auto` (the default) trusts the model's own declared context window uncapped — this is deliberate, so a large-context model like `Qwen3-Embedding-4B` isn't silently restricted to 512 tokens. But requests are tokenized with `padding=True`, so every text in a batch is padded to the length of the *longest* text in that same batch: one long outlier drags the whole batch up to its length, not just that one input. Combined with attention cost scaling roughly quadratically with sequence length, an unbounded large-context model can use dramatically more memory per batch than the 512-token models this server shipped with historically.

If you pin a large-context embedding model with `--embedding-model`, the server logs a `WARNING` at startup when the resolved context exceeds 4096 tokens and no `--embedding-max-length` is set, as a nudge to set an explicit, memory-appropriate ceiling for your hardware (e.g. `--embedding-max-length 4096`) rather than running fully unbounded in production.

Example `error`-policy response when an input exceeds the effective limit:

```json
{
  "detail": {
    "error": "embedding_input_too_long",
    "message": "Input 0 has 1400 tokens, exceeding the effective embedding max length of 1024",
    "input_index": 0,
    "token_count": 1400,
    "max_length": 1024
  }
}
```

The effective embedding `max_length` and `overflow_policy` are also reported under the `embedding` key of `GET /v1/status`.

## API Reference

### POST /v1/embeddings

Create embeddings for the given input text(s).

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Supported embedding model ID, or the startup-pinned model when `--embedding-model` is used |
| `input` | string or list[string] | Yes | Text(s) to embed |

**Response:**

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

A 400 response with a structured `embedding_input_too_long` detail (see above) is returned when `--embedding-overflow-policy error` is set and an input exceeds the effective max length.

## Python API

### Direct usage without server

```python
from vllm_mlx.embedding import EmbeddingEngine

engine = EmbeddingEngine("mlx-community/all-MiniLM-L6-v2-4bit")
engine.load()

vectors = engine.embed(["Hello world", "How are you?"])
print(f"Dimensions: {len(vectors[0])}")

tokens = engine.count_tokens(["Hello world"])
print(f"Token count: {tokens}")
```

## Troubleshooting

### mlx-embeddings not installed

```
pip install mlx-embeddings>=0.0.5
```

### Model not found

Make sure the model name matches one of the supported request-time IDs above, or start the server with `--embedding-model` to pin a custom model. You can pre-download supported models:

```bash
huggingface-cli download mlx-community/all-MiniLM-L6-v2-4bit
```
