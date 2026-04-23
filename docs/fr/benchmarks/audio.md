# Benchmarks Audio

## Benchmarks STT (Speech-to-Text)

### Lancer les benchmarks STT

```bash
# Run with default test audio
python examples/benchmark_audio.py --stt

# Run with your own audio file
python examples/benchmark_audio.py --stt --audio path/to/audio.wav
```

### Résultats (M4 Max, 128 Go)

**Audio de test :** 46,7 secondes de synthèse vocale

| Model | Parameters | Load Time | Transcribe Time | RTF* |
|-------|------------|-----------|-----------------|------|
| whisper-tiny | 39M | 0.34s | 0.24s | **197x** |
| whisper-small | 244M | 0.18s | 0.47s | **98x** |
| whisper-medium | 769M | 0.35s | 1.15s | **41x** |
| whisper-large-v3 | 1.5B | 0.50s | 1.96s | **24x** |
| whisper-large-v3-turbo | 809M | 0.12s | 0.86s | **55x** |

*RTF = Real-Time Factor (plus la valeur est élevée, plus c'est rapide). Un RTF de 100x signifie qu'une minute d'audio est transcrite en environ 0,6 secondes.*

### Résultats (M1 Max, 64 Go)

STT avec Parakeet (environnement par défaut, Whisper indisponible en raison d'une incompatibilité de dépendance numpy) :

| Model | Load Time | Transcribe Time | RTF |
|-------|-----------|-----------------|-----|
| parakeet-tdt-0.6b-v2 | 0.28s | 1.01s | **9.9x** |
| parakeet-tdt-0.6b-v3 | 0.30s | 0.19s | **52.7x** |

STT avec Whisper (`numpy==2.3.5` explicite + `uv run --no-sync`) :

| Model | Load Time | Transcribe Time | RTF |
|-------|-----------|-----------------|-----|
| whisper-tiny | 4.02s | 1.05s | **9.5x** |
| whisper-small | 10.15s | 1.03s | **9.7x** |
| whisper-medium | 22.96s | 2.20s | **4.6x** |
| whisper-large-v3 | 38.34s | 0.96s | **10.5x** |
| whisper-large-v3-turbo | 21.79s | 0.70s | **14.3x** |
| parakeet-tdt-0.6b-v2 | 0.47s | 0.18s | **54.4x** |
| parakeet-tdt-0.6b-v3 | 1.13s | 0.18s | **54.6x** |

### Recommandations de modèles

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Transcription en temps réel** | whisper-tiny | Le plus rapide (RTF 197x), faible latence |
| **Usage général** | whisper-large-v3-turbo | Meilleur compromis vitesse (55x) et qualité |
| **Précision maximale** | whisper-large-v3 | Le plus précis, prend en charge plus de 99 langues |
| **Mémoire limitée** | whisper-small | Bonne qualité à 244M paramètres |

### Qualité de transcription

Tous les modèles ont correctement transcrit l'audio de test. Exemple de sortie :

```
Input text:
"Welcome to this comprehensive speech to text demonstration.
This audio sample is designed to test the accuracy and speed of various speech recognition models.
The quick brown fox jumps over the lazy dog..."

Whisper-large-v3 output:
"Welcome to this comprehensive speech to text demonstration.
This audio sample is designed to test the accuracy and speed of various speech recognition models.
The quick brown fox jumps over the lazy dog..." (identical)
```

### Langues prises en charge

Les modèles Whisper prennent en charge plus de 99 langues, notamment :
- Anglais, espagnol, français, allemand, italien, portugais
- Chinois (mandarin, cantonais), japonais, coréen
- Arabe, hindi, russe, turc, ukrainien
- Et bien d'autres

## Benchmarks TTS (Text-to-Speech)

### Lancer les benchmarks TTS

```bash
python examples/benchmark_audio.py --tts
```

### Résultats (M4 Max, 128 Go)

**Test :** Génération audio pour 3 échantillons de texte (court, moyen, long)

| Model | Load Time | Chars/sec | RTF* |
|-------|-----------|-----------|------|
| Kokoro-82M-bf16 | 0.8s | 350+ | **22x** |
| Kokoro-82M-4bit | 0.4s | 320+ | **20x** |

*RTF = Real-Time Factor. Un RTF de 22x signifie qu'une seconde d'audio est générée en environ 0,045 secondes.*

### Résultats TTS (M1 Max, 64 Go)

| Model | Load Time | Avg Chars/s | Avg RTF |
|-------|-----------|-------------|---------|
| Kokoro-82M-bf16 | 2.81s | 176.0 | **11.9x** |
| Kokoro-82M-4bit | 0.22s | 225.6 | **15.5x** |

### Qualité TTS

Kokoro produit une synthèse vocale au son naturel avec :
- 11 voix intégrées (masculines et féminines)
- Prise en charge de 8 langues (anglais, espagnol, français, japonais, chinois, italien, portugais, hindi)
- 82M paramètres, rapide et léger

## Benchmarks de traitement audio

### SAM-Audio (séparation de sources)

**Test :** Séparation de la batterie dans un morceau de rock de 30 secondes

| Metric | Value |
|--------|-------|
| Model | sam-audio-large-fp16 |
| Processing time | ~20s |
| Peak memory | ~27 GB |
| Output sample rate | 48000 Hz |

## Lancer tous les benchmarks audio

```bash
# Run all benchmarks
python examples/benchmark_audio.py --all

# Or run individually
python examples/benchmark_audio.py --stt
python examples/benchmark_audio.py --tts
```

## Modèles disponibles sur mlx-community

### Modèles STT
- `mlx-community/whisper-tiny-mlx`
- `mlx-community/whisper-small-mlx`
- `mlx-community/whisper-medium-mlx`
- `mlx-community/whisper-large-v3-mlx`
- `mlx-community/whisper-large-v3-turbo`
- `mlx-community/parakeet-tdt-0.6b-v2`
- `mlx-community/parakeet-tdt-0.6b-v3`

### Modèles TTS
- `mlx-community/Kokoro-82M-bf16` (recommandé)
- `mlx-community/Kokoro-82M-4bit`
- `mlx-community/chatterbox-turbo-fp16`
- `mlx-community/VibeVoice-Realtime-0.5B-4bit`

### Traitement audio
- `mlx-community/sam-audio-large-fp16`
