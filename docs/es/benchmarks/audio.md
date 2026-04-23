# Benchmarks de Audio

## Benchmarks de Speech-to-Text (STT)

### Ejecutar benchmarks de STT

```bash
# Run with default test audio
python examples/benchmark_audio.py --stt

# Run with your own audio file
python examples/benchmark_audio.py --stt --audio path/to/audio.wav
```

### Resultados (M4 Max, 128GB)

**Audio de prueba:** 46.7 segundos de voz sintetizada

| Model | Parameters | Load Time | Transcribe Time | RTF* |
|-------|------------|-----------|-----------------|------|
| whisper-tiny | 39M | 0.34s | 0.24s | **197x** |
| whisper-small | 244M | 0.18s | 0.47s | **98x** |
| whisper-medium | 769M | 0.35s | 1.15s | **41x** |
| whisper-large-v3 | 1.5B | 0.50s | 1.96s | **24x** |
| whisper-large-v3-turbo | 809M | 0.12s | 0.86s | **55x** |

*RTF = Real-Time Factor (mayor es más rápido). Un RTF de 100x significa que 1 minuto de audio se transcribe en aprox. 0.6 segundos.*

### Resultados (M1 Max, 64GB)

STT con Parakeet (entorno predeterminado, Whisper no disponible por incompatibilidad de dependencia con numpy):

| Model | Load Time | Transcribe Time | RTF |
|-------|-----------|-----------------|-----|
| parakeet-tdt-0.6b-v2 | 0.28s | 1.01s | **9.9x** |
| parakeet-tdt-0.6b-v3 | 0.30s | 0.19s | **52.7x** |

STT con Whisper (`numpy==2.3.5` explícito + `uv run --no-sync`):

| Model | Load Time | Transcribe Time | RTF |
|-------|-----------|-----------------|-----|
| whisper-tiny | 4.02s | 1.05s | **9.5x** |
| whisper-small | 10.15s | 1.03s | **9.7x** |
| whisper-medium | 22.96s | 2.20s | **4.6x** |
| whisper-large-v3 | 38.34s | 0.96s | **10.5x** |
| whisper-large-v3-turbo | 21.79s | 0.70s | **14.3x** |
| parakeet-tdt-0.6b-v2 | 0.47s | 0.18s | **54.4x** |
| parakeet-tdt-0.6b-v3 | 1.13s | 0.18s | **54.6x** |

### Recomendaciones de modelos

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Transcripcion en tiempo real** | whisper-tiny | El más rápido (197x RTF), baja latencia |
| **Uso general** | whisper-large-v3-turbo | Mejor equilibrio entre velocidad (55x) y calidad |
| **Mayor precision** | whisper-large-v3 | El más preciso, soporta más de 99 idiomas |
| **Memoria reducida** | whisper-small | Buena calidad con 244M parámetros |

### Calidad de transcripción

Todos los modelos transcribieron correctamente el audio de prueba. Ejemplo de salida:

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

### Idiomas soportados

Los modelos Whisper soportan más de 99 idiomas, entre ellos:
- Inglés, español, francés, alemán, italiano, portugués
- Chino (mandarín, cantonés), japonés, coreano
- Árabe, hindi, ruso, turco, ucraniano
- Y muchos más

## Benchmarks de Text-to-Speech (TTS)

### Ejecutar benchmarks de TTS

```bash
python examples/benchmark_audio.py --tts
```

### Resultados (M4 Max, 128GB)

**Prueba:** Generar audio para 3 muestras de texto (corta, media, larga)

| Model | Load Time | Chars/sec | RTF* |
|-------|-----------|-----------|------|
| Kokoro-82M-bf16 | 0.8s | 350+ | **22x** |
| Kokoro-82M-4bit | 0.4s | 320+ | **20x** |

*RTF = Real-Time Factor. Un RTF de 22x significa que 1 segundo de audio se genera en aprox. 0.045 segundos.*

### Resultados de TTS (M1 Max, 64GB)

| Model | Load Time | Avg Chars/s | Avg RTF |
|-------|-----------|-------------|---------|
| Kokoro-82M-bf16 | 2.81s | 176.0 | **11.9x** |
| Kokoro-82M-4bit | 0.22s | 225.6 | **15.5x** |

### Calidad de TTS

Kokoro produce voz con sonido natural, con:
- 11 voces integradas (masculinas y femeninas)
- Soporte para 8 idiomas (inglés, español, francés, japonés, chino, italiano, portugués, hindi)
- 82M parámetros, rápido y liviano

## Benchmarks de procesamiento de audio

### SAM-Audio (separacion de fuentes)

**Prueba:** Separar la bateria de una cancion de rock de 30 segundos

| Metric | Value |
|--------|-------|
| Model | sam-audio-large-fp16 |
| Processing time | ~20s |
| Peak memory | ~27 GB |
| Output sample rate | 48000 Hz |

## Ejecutar todos los benchmarks de audio

```bash
# Run all benchmarks
python examples/benchmark_audio.py --all

# Or run individually
python examples/benchmark_audio.py --stt
python examples/benchmark_audio.py --tts
```

## Modelos disponibles en mlx-community

### Modelos STT
- `mlx-community/whisper-tiny-mlx`
- `mlx-community/whisper-small-mlx`
- `mlx-community/whisper-medium-mlx`
- `mlx-community/whisper-large-v3-mlx`
- `mlx-community/whisper-large-v3-turbo`
- `mlx-community/parakeet-tdt-0.6b-v2`
- `mlx-community/parakeet-tdt-0.6b-v3`

### Modelos TTS
- `mlx-community/Kokoro-82M-bf16` (recommended)
- `mlx-community/Kokoro-82M-4bit`
- `mlx-community/chatterbox-turbo-fp16`
- `mlx-community/VibeVoice-Realtime-0.5B-4bit`

### Procesamiento de audio
- `mlx-community/sam-audio-large-fp16`
