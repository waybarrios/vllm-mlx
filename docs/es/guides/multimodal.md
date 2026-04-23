# Modelos Multimodales (Imágenes y Video)

vllm-mlx soporta modelos de visión y lenguaje (VLM) para el análisis de imágenes y video.

## Modelos Soportados

- Qwen3-VL (recomendado)
- Qwen2-VL
- Gemma 3
- LLaVA
- Idefics
- PaliGemma
- Pixtral
- Molmo
- DeepSeek-VL

## Iniciar un Servidor Multimodal

```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

Los modelos que contienen "VL", "Vision" o "mllm" en el nombre se detectan automáticamente como multimodales.

## Análisis de Imágenes

### Mediante el SDK de OpenAI

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Imagen desde URL
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

### Imágenes en Base64

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

### Mediante curl

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

## Análisis de Video

### Mediante el SDK de OpenAI

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

### Parámetros de Video

Controla la extracción de fotogramas mediante parámetros adicionales en el cuerpo de la solicitud:

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

### Mediante curl

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

## Formatos Soportados

### Imágenes

| Formato | Ejemplo |
|--------|---------|
| URL | `{"type": "image_url", "image_url": {"url": "https://..."}}` |
| Archivo local | `{"type": "image_url", "image_url": {"url": "/path/to/image.jpg"}}` |
| Base64 | `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}` |

### Videos

| Formato | Ejemplo |
|--------|---------|
| URL | `{"type": "video_url", "video_url": {"url": "https://..."}}` |
| Archivo local | `{"type": "video", "video": "/path/to/video.mp4"}` |
| Base64 | `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}` |

## API de Python

```python
from vllm_mlx.models import MLXMultimodalLM

mllm = MLXMultimodalLM("mlx-community/Qwen3-VL-4B-Instruct-3bit")
mllm.load()

# Imagen
description = mllm.describe_image("photo.jpg")

# Video
description = mllm.describe_video("video.mp4", fps=2.0)

# Prompt personalizado
output = mllm.generate(
    prompt="Compare these images",
    images=["img1.jpg", "img2.jpg"]
)
```

## Consejos de Rendimiento

### Imágenes
- Las resoluciones menores se procesan más rápido (224x224 vs 1920x1080)
- Usa la resolución adecuada para tu tarea

### Videos
- Menor FPS = procesamiento más rápido
- Menos fotogramas = menor uso de memoria
- 64 fotogramas es el máximo práctico (96 o más causa timeout en la GPU)

## Benchmarks

Probado en Apple M4 Max con 128 GB de memoria unificada.

### Qwen3-VL-4B-Instruct-3bit

| Resolución | Tiempo | Tokens | Velocidad | Memoria |
|------------|------|--------|-------|--------|
| 224x224 | 0.87s | 124 | 143 tok/s | 2.6 GB |
| 448x448 | 1.01s | 107 | 106 tok/s | 3.1 GB |
| 768x768 | 1.42s | 127 | 89 tok/s | 3.4 GB |
| 1024x1024 | 1.85s | 116 | 63 tok/s | 3.6 GB |

### Qwen3-VL-8B-Instruct-4bit

| Resolución | Tiempo | Tokens | Velocidad | Memoria |
|------------|------|--------|-------|--------|
| 224x224 | 1.08s | 78 | 73 tok/s | 5.6 GB |
| 448x448 | 1.41s | 70 | 50 tok/s | 6.1 GB |
| 768x768 | 2.06s | 91 | 44 tok/s | 6.5 GB |
| 1024x1024 | 3.02s | 76 | 25 tok/s | 7.6 GB |

### Gemma 3 4B 4bit

| Resolución | Tiempo | Tokens | Velocidad | Memoria |
|------------|------|--------|-------|--------|
| 224x224 | 0.95s | 30 | 32 tok/s | 5.2 GB |
| 448x448 | 0.99s | 34 | 34 tok/s | 5.2 GB |
| 768x768 | 0.99s | 32 | 32 tok/s | 5.2 GB |
| 1024x1024 | 0.95s | 28 | 29 tok/s | 5.2 GB |

### Ejecutar Benchmarks

```bash
# Benchmark rápido
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --quick

# Benchmark completo con más resoluciones
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit

# Benchmark de video
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --video
```

## MLLM Cache

vllm-mlx incluye un sistema de prefix cache para modelos multimodales que puede acelerar significativamente las solicitudes repetidas con las mismas imágenes.

### Cómo Funciona

Cuando envías una imagen al modelo, el encoder de visión la procesa y genera embeddings. Este procesamiento toma entre 1 y 2 segundos. El MLLM cache almacena esos embeddings junto con el estado del KV cache, de modo que las solicitudes posteriores con la misma imagen omiten el encoder de visión por completo.

El cache utiliza hashing basado en contenido (similar a LMCache) para identificar imágenes idénticas sin importar cómo se proporcionen (URL, base64 o ruta de archivo).

### Habilitar el Cache

```bash
# Habilitar con configuración predeterminada (512 MB máximo)
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --enable-mllm-cache

# Con límite de memoria personalizado
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit \
    --enable-mllm-cache \
    --mllm-cache-max-mb 1024
```

### API de Python

```python
from vllm_mlx.mllm_cache import MLLMPrefixCacheManager

# Crear el gestor de cache
cache = MLLMPrefixCacheManager(max_memory_mb=512)

# Almacenar embeddings y KV cache tras el procesamiento
cache.store(
    images=["photo.jpg"],
    prompt="Describe this image",
    vision_embeddings=embeddings,
    kv_cache=kv_state,
    num_tokens=128
)

# Recuperar del cache en solicitudes posteriores
entry, match_len = cache.fetch(images=["photo.jpg"], prompt="Describe this image")
if entry:
    # Usar embeddings en cache, omitir el encoder de visión
    embeddings = entry.vision_embeddings
    kv_state = entry.kv_cache
```

### Estadísticas del Cache

```python
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Memory used: {stats.memory_used_mb:.1f} MB")
print(f"Tokens saved: {stats.tokens_saved}")
```

### Gestión de Memoria

El cache utiliza evicción LRU (Least Recently Used) cuando se alcanza el límite de memoria. Cada entrada registra:

- Tamaño de los embeddings de visión
- Tamaño del KV cache por capa
- Frecuencia de acceso para el ordenamiento LRU

Cuando hay presión de memoria, las entradas con acceso menos reciente se evictan primero.

## Gradio Chat UI

Para chat multimodal interactivo:

```bash
vllm-mlx-chat --served-model-name mlx-community/Qwen3-VL-4B-Instruct-3bit
```

Soporta arrastrar y soltar imágenes y videos.
