# Python API

API de Python directa para acceso programático a vllm-mlx.

## Modelos de lenguaje

### Uso básico

```python
from vllm_mlx.models import MLXLanguageModel

# Load model
model = MLXLanguageModel("mlx-community/Llama-3.2-3B-Instruct-4bit")
model.load()

# Generate text
output = model.generate("What is the capital of France?", max_tokens=100)
print(output.text)
```

### Generación con streaming

```python
for chunk in model.stream_generate("Tell me a story about a robot"):
    print(chunk.text, end="", flush=True)
```

### Interfaz de chat

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, who are you?"}
]
response = model.chat(messages)
print(response.text)
```

### Parámetros de generación

```python
output = model.generate(
    prompt="Write a poem",
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    stop=["END", "\n\n"]
)
```

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|---------|
| `max_tokens` | Cantidad máxima de tokens a generar | 256 |
| `temperature` | Temperatura de muestreo (0-2) | 0.7 |
| `top_p` | Nucleus sampling | 0.9 |
| `stop` | Secuencias de parada | None |

## Modelos de visión y lenguaje (VLM)

### Uso básico

```python
from vllm_mlx.models import MLXMultimodalLM

# Load model
mllm = MLXMultimodalLM("mlx-community/Qwen3-VL-4B-Instruct-3bit")
mllm.load()

# Describe an image
description = mllm.describe_image("photo.jpg")
print(description)
```

### Preguntas sobre imágenes

```python
answer = mllm.answer_about_image("photo.jpg", "What color is the car?")
print(answer)
```

### Varias imágenes

```python
output = mllm.generate(
    prompt="Compare these two images",
    images=["image1.jpg", "image2.jpg"]
)
print(output.text)
```

### Comprensión de video

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

### Parámetros de video

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|---------|
| `video_fps` | Fotogramas por segundo a extraer | 2.0 |
| `video_max_frames` | Cantidad máxima de fotogramas a procesar | 32 |

## Engine API

Para casos de uso avanzados, se puede usar el engine directamente:

### Engine simple

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

### Engine con batching

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

## Formato de salida

Todos los métodos de generación retornan un objeto `GenerationOutput`:

```python
output = model.generate("Hello")

print(output.text)              # Generated text
print(output.prompt_tokens)     # Input token count
print(output.completion_tokens) # Output token count
print(output.finish_reason)     # "stop" or "length"
```

## Manejo de errores

```python
from vllm_mlx.models import MLXLanguageModel

try:
    model = MLXLanguageModel("invalid-model")
    model.load()
except Exception as e:
    print(f"Failed to load model: {e}")
```
