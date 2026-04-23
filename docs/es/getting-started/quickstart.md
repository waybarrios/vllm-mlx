# Inicio rápido

## Opción 1: Servidor compatible con OpenAI

Inicia el servidor:

```bash
# Simple mode - maximum throughput for single user
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# Continuous batching - for multiple concurrent users
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

Úsalo con el SDK de Python de OpenAI:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

O con curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Opción 2: API de Python directa

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

## Opción 3: Interfaz de chat con Gradio

```bash
vllm-mlx-chat --served-model-name mlx-community/Llama-3.2-3B-Instruct-4bit
```

Abre una interfaz web en http://localhost:7860

## Modelos multimodales

Para comprensión de imágenes y video, usa un modelo VLM:

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

## Modelos de razonamiento

Separa el proceso de pensamiento del modelo de la respuesta final:

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

Genera embeddings de texto para búsqueda semántica y RAG:

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

Habilita la llamada a funciones con cualquier modelo compatible:

```bash
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral
```

## Próximos pasos

- [Guía del servidor](../guides/server.md) - Configuración completa del servidor
- [API de Python](../guides/python-api.md) - Uso directo de la API
- [Guía multimodal](../guides/multimodal.md) - Imágenes y video
- [Guía de audio](../guides/audio.md) - Speech-to-Text y Text-to-Speech
- [Guía de embeddings](../guides/embeddings.md) - Embeddings de texto
- [Modelos de razonamiento](../guides/reasoning.md) - Modelos con pensamiento
- [Tool Calling](../guides/tool-calling.md) - Llamada a funciones
- [Modelos compatibles](../reference/models.md) - Modelos disponibles
