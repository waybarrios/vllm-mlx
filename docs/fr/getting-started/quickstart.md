# Démarrage rapide

## Option 1 : serveur compatible OpenAI

Démarrez le serveur :

```bash
# Simple mode - maximum throughput for single user
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# Continuous batching - for multiple concurrent users
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

Utilisation avec le SDK Python OpenAI :

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Ou avec curl :

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Option 2 : API Python directe

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

## Option 3 : interface de chat Gradio

```bash
vllm-mlx-chat --served-model-name mlx-community/Llama-3.2-3B-Instruct-4bit
```

Ouvre une interface web à l'adresse http://localhost:7860

## Modèles multimodaux

Pour la compréhension d'images et de vidéos, utilisez un modèle VLM :

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

## Modèles de raisonnement

Séparez le processus de réflexion du modèle de la réponse finale :

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

Générez des embeddings textuels pour la recherche sémantique et le RAG :

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

Activez l'appel de fonctions avec tout modèle compatible :

```bash
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral
```

## Étapes suivantes

- [Server Guide](../guides/server.md) - Configuration complète du serveur
- [Python API](../guides/python-api.md) - Utilisation directe de l'API
- [Multimodal Guide](../guides/multimodal.md) - Images et vidéos
- [Audio Guide](../guides/audio.md) - Speech-to-Text et Text-to-Speech
- [Embeddings Guide](../guides/embeddings.md) - Embeddings textuels
- [Reasoning Models](../guides/reasoning.md) - Modèles de réflexion
- [Tool Calling](../guides/tool-calling.md) - Appel de fonctions
- [Supported Models](../reference/models.md) - Modèles disponibles
