# Python API

API Python directe pour un accès programmatique à vllm-mlx.

## Modèles de langage

### Utilisation de base

```python
from vllm_mlx.models import MLXLanguageModel

# Load model
model = MLXLanguageModel("mlx-community/Llama-3.2-3B-Instruct-4bit")
model.load()

# Generate text
output = model.generate("What is the capital of France?", max_tokens=100)
print(output.text)
```

### Génération en streaming

```python
for chunk in model.stream_generate("Tell me a story about a robot"):
    print(chunk.text, end="", flush=True)
```

### Interface de chat

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, who are you?"}
]
response = model.chat(messages)
print(response.text)
```

### Paramètres de génération

```python
output = model.generate(
    prompt="Write a poem",
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    stop=["END", "\n\n"]
)
```

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `max_tokens` | Nombre maximum de tokens à générer | 256 |
| `temperature` | Température d'échantillonnage (0-2) | 0.7 |
| `top_p` | Nucleus sampling | 0.9 |
| `stop` | Séquences d'arrêt | None |

## Modèles vision-langage

### Utilisation de base

```python
from vllm_mlx.models import MLXMultimodalLM

# Load model
mllm = MLXMultimodalLM("mlx-community/Qwen3-VL-4B-Instruct-3bit")
mllm.load()

# Describe an image
description = mllm.describe_image("photo.jpg")
print(description)
```

### Questions-réponses sur une image

```python
answer = mllm.answer_about_image("photo.jpg", "What color is the car?")
print(answer)
```

### Plusieurs images

```python
output = mllm.generate(
    prompt="Compare these two images",
    images=["image1.jpg", "image2.jpg"]
)
print(output.text)
```

### Compréhension vidéo

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

### Paramètres vidéo

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `video_fps` | Images par seconde à extraire | 2.0 |
| `video_max_frames` | Nombre maximum d'images à traiter | 32 |

## API du moteur

Pour les cas d'utilisation avancés, utilisez le moteur directement :

### Moteur simple

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

### Moteur avec batching

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

## Format de sortie

Toutes les méthodes de génération retournent un objet `GenerationOutput` :

```python
output = model.generate("Hello")

print(output.text)              # Generated text
print(output.prompt_tokens)     # Input token count
print(output.completion_tokens) # Output token count
print(output.finish_reason)     # "stop" or "length"
```

## Gestion des erreurs

```python
from vllm_mlx.models import MLXLanguageModel

try:
    model = MLXLanguageModel("invalid-model")
    model.load()
except Exception as e:
    print(f"Failed to load model: {e}")
```
