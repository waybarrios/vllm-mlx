# Quick Start

## First response

Complete the [isolated release install](installation.md#install-a-release)
first. Keep its environment activated. This starts one small text model with
the default engine; advanced features and coding clients come afterward.
The first launch downloads the model if it is not already cached. Keep this
terminal open until the server reports it is ready.

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --host 127.0.0.1 --port 8000
```

In a second terminal, check readiness and the served model before sending work:

```bash
curl --fail --show-error http://127.0.0.1:8000/health
curl --fail --show-error http://127.0.0.1:8000/v1/models
curl --fail --show-error http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Llama-3.2-3B-Instruct-4bit","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
```

Success is a nonempty answer in `choices[0].message.content`, not just a
listening port. Check `/v1/models` again if a client reports a model mismatch.
If port 8000 is occupied, choose a free port and change it in every URL; do not
terminate an unfamiliar process. Keep the host on loopback for local use.
Stop the server with Ctrl-C before starting a different model.

The minimal command uses the default engine without continuous batching.
After the first response, stop it and add `--continuous-batching` to the same
serve command to try that engine. See the
[continuous-batching guide](../guides/continuous-batching.md) for its cache
options and supported configurations.

If a CLI flag is unrecognized, check `vllm-mlx serve --help` in the same
environment and the installed release. Do not copy options from newer source
documentation into an older wheel. Download/authentication failures belong to
model acquisition; unsupported architecture/parser errors require a supported
model and dependency combination, not guessed flags. Include the exact model
repository/revision, command, server error and package versions in a report.

## Artifact and profile choices

Choose an artifact first, then settings supported by that artifact and the
installed server version. The example above uses a community 4-bit MLX
conversion, not the vendor's original weights.

| Choice | What to preserve and disclose |
|---|---|
| Vendor-documented artifact/settings | Exact vendor revision, tokenizer, template, sampling and architecture requirements. Confirm that the artifact format and model implementation are supported by MLX; a vendor CUDA command is not a Mac launch recipe. |
| Lower-memory converted artifact | Conversion source/revision, quantization and any changed storage method, plus the tested context and feature limits. Label it as a derivative; do not claim weight or quality equivalence to the vendor artifact. |

Weight quantization, KV-cache quantization, and SSD prefix-cache persistence
are different choices. A prefix cache on SSD does not make model weights
SSD-streamed. External quantized embedding storage likewise does not establish
support for generic expert offloading or a larger context. Use only documented
combinations; MTP, SpecPrefill and cache features are not universally composable.
Leave untested combinations off rather than inheriting another model's flags.

After the first response, use the existing SDK or chat UI examples below.
The [coding-client validation work](https://github.com/waybarrios/vllm-mlx/pull/774)
tracks client-specific setup and evidence; check its merge/release status
before treating those results as coverage for your installation.
Test one client and its required tool/reasoning contract before
adding concurrent workloads. Basic chat success alone does not qualify tools,
vision, structured output or long context.

## Option 1: OpenAI-Compatible Server

Start the server:

```bash
# Simple mode
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# Continuous batching - for multiple concurrent users
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

Use with OpenAI Python SDK:

Leave the server running. In another terminal, install the SDK in your Python
client environment (the server package does not install `openai`):

```bash
python -m pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Or with curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Llama-3.2-3B-Instruct-4bit", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Option 2: Direct Python API

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

## Option 3: Gradio Chat UI

```bash
vllm-mlx-chat --served-model-name mlx-community/Llama-3.2-3B-Instruct-4bit
```

Opens a web interface at http://localhost:7860

## Multimodal Models

For image/video understanding, use a VLM model:

```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

```python
response = client.chat.completions.create(
    model="mlx-community/Qwen3-VL-4B-Instruct-3bit",
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

## Reasoning Models

Separate the model's thinking process from the final answer:

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

```python
response = client.chat.completions.create(
    model="mlx-community/Qwen3-8B-4bit",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)
print(response.choices[0].message.content)  # Final answer
```

## Embeddings

Generate text embeddings for semantic search and RAG:

```bash
vllm-mlx serve mlx-community/Qwen3-4B-4bit --embedding-model mlx-community/multilingual-e5-small-mlx
```

```python
response = client.embeddings.create(
    model="mlx-community/multilingual-e5-small-mlx",
    input="Hello world"
)
```

## Tool Calling

Enable function calling with any supported model:

```bash
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral
```

## Next Steps

- [Server Guide](../guides/server.md) - Full server configuration
- [Python API](../guides/python-api.md) - Direct API usage
- [Multimodal Guide](../guides/multimodal.md) - Images and video
- [Audio Guide](../guides/audio.md) - Speech-to-Text and Text-to-Speech
- [Embeddings Guide](../guides/embeddings.md) - Text embeddings
- [Reasoning Models](../guides/reasoning.md) - Thinking models
- [Tool Calling](../guides/tool-calling.md) - Function calling
- [Supported Models](../reference/models.md) - Available models
