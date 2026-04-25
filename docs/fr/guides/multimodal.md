# Modèles multimodaux (images et vidéos)

vllm-mlx prend en charge les VLM pour la compréhension des images et des vidéos.

## Modèles pris en charge

- Qwen3-VL (recommandé)
- Qwen2-VL
- Gemma 3
- LLaVA
- Idefics
- PaliGemma
- Pixtral
- Molmo
- DeepSeek-VL

## Démarrer un serveur multimodal

```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

Les modèles dont le nom contient « VL », « Vision » ou « mllm » sont automatiquement détectés comme multimodaux.

## Analyse d'images

### Via le SDK OpenAI

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Image depuis une URL
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

### Images en Base64

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

### Via curl

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

## Analyse de vidéos

### Via le SDK OpenAI

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

### Paramètres vidéo

Contrôlez l'extraction des images via les paramètres du corps étendu :

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

### Via curl

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

## Formats pris en charge

### Images

| Format | Exemple |
|--------|---------|
| URL | `{"type": "image_url", "image_url": {"url": "https://..."}}` |
| Fichier local | `{"type": "image_url", "image_url": {"url": "/path/to/image.jpg"}}` |
| Base64 | `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}` |

### Vidéos

| Format | Exemple |
|--------|---------|
| URL | `{"type": "video_url", "video_url": {"url": "https://..."}}` |
| Fichier local | `{"type": "video", "video": "/path/to/video.mp4"}` |
| Base64 | `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}` |

## API Python

```python
from vllm_mlx.models import MLXMultimodalLM

mllm = MLXMultimodalLM("mlx-community/Qwen3-VL-4B-Instruct-3bit")
mllm.load()

# Image
description = mllm.describe_image("photo.jpg")

# Vidéo
description = mllm.describe_video("video.mp4", fps=2.0)

# Prompt personnalisé
output = mllm.generate(
    prompt="Compare these images",
    images=["img1.jpg", "img2.jpg"]
)
```

## Conseils de performance

### Images
- Les résolutions plus petites sont traitées plus rapidement (224x224 vs 1920x1080)
- Utilisez la résolution adaptée à votre tâche

### Vidéos
- Un FPS plus bas accélère le traitement
- Moins d'images signifie moins de mémoire utilisée
- 64 images est le maximum pratique (96 et plus provoque un timeout GPU)

## Benchmarks

Testés sur Apple M4 Max avec 128 Go de mémoire unifiée.

### Qwen3-VL-4B-Instruct-3bit

| Résolution | Temps | Tokens | Vitesse | Mémoire |
|------------|-------|--------|---------|---------|
| 224x224 | 0.87s | 124 | 143 tok/s | 2.6 Go |
| 448x448 | 1.01s | 107 | 106 tok/s | 3.1 Go |
| 768x768 | 1.42s | 127 | 89 tok/s | 3.4 Go |
| 1024x1024 | 1.85s | 116 | 63 tok/s | 3.6 Go |

### Qwen3-VL-8B-Instruct-4bit

| Résolution | Temps | Tokens | Vitesse | Mémoire |
|------------|-------|--------|---------|---------|
| 224x224 | 1.08s | 78 | 73 tok/s | 5.6 Go |
| 448x448 | 1.41s | 70 | 50 tok/s | 6.1 Go |
| 768x768 | 2.06s | 91 | 44 tok/s | 6.5 Go |
| 1024x1024 | 3.02s | 76 | 25 tok/s | 7.6 Go |

### Gemma 3 4B 4bit

| Résolution | Temps | Tokens | Vitesse | Mémoire |
|------------|-------|--------|---------|---------|
| 224x224 | 0.95s | 30 | 32 tok/s | 5.2 Go |
| 448x448 | 0.99s | 34 | 34 tok/s | 5.2 Go |
| 768x768 | 0.99s | 32 | 32 tok/s | 5.2 Go |
| 1024x1024 | 0.95s | 28 | 29 tok/s | 5.2 Go |

### Lancer les benchmarks

```bash
# Benchmark rapide
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --quick

# Benchmark complet avec plus de résolutions
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit

# Benchmark vidéo
vllm-mlx-bench --model mlx-community/Qwen3-VL-4B-Instruct-3bit --video
```

## Cache MLLM

vllm-mlx inclut un système de prefix cache pour les modèles multimodaux, capable d'accélérer significativement les requêtes répétées utilisant les mêmes images.

### Fonctionnement

Lorsque vous envoyez une image au modèle, l'encodeur de vision la traite en embeddings. Ce traitement prend 1 à 2 secondes. Le cache MLLM stocke ces embeddings ainsi que l'état du KV cache, de sorte que les requêtes ultérieures avec la même image contournent entièrement l'encodeur de vision.

Le cache utilise un hachage basé sur le contenu (similaire à LMCache) pour identifier les images identiques, quelle que soit leur forme de transmission (URL, base64 ou chemin de fichier).

### Activer le cache

```bash
# Activer avec les paramètres par défaut (512 Mo maximum)
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --enable-mllm-cache

# Avec une limite mémoire personnalisée
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit \
    --enable-mllm-cache \
    --mllm-cache-max-mb 1024
```

### API Python

```python
from vllm_mlx.mllm_cache import MLLMPrefixCacheManager

# Créer le gestionnaire de cache
cache = MLLMPrefixCacheManager(max_memory_mb=512)

# Stocker les embeddings et le KV cache après traitement
cache.store(
    images=["photo.jpg"],
    prompt="Describe this image",
    vision_embeddings=embeddings,
    kv_cache=kv_state,
    num_tokens=128
)

# Récupérer depuis le cache lors des requêtes suivantes
entry, match_len = cache.fetch(images=["photo.jpg"], prompt="Describe this image")
if entry:
    # Utiliser les embeddings mis en cache, contourner l'encodeur de vision
    embeddings = entry.vision_embeddings
    kv_state = entry.kv_cache
```

### Statistiques du cache

```python
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Memory used: {stats.memory_used_mb:.1f} MB")
print(f"Tokens saved: {stats.tokens_saved}")
```

### Gestion de la mémoire

Le cache utilise une éviction LRU (Least Recently Used) lorsque la limite mémoire est atteinte. Chaque entrée suit :

- La taille des embeddings de vision
- La taille du KV cache par couche
- La fréquence d'accès pour l'ordonnancement LRU

En cas de pression mémoire, les entrées les moins récemment consultées sont évincées en premier.

## Interface de chat Gradio

Pour un chat multimodal interactif :

```bash
vllm-mlx-chat --served-model-name mlx-community/Qwen3-VL-4B-Instruct-3bit
```

Prend en charge le glisser-déposer d'images et de vidéos.
