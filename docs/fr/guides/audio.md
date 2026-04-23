# Support Audio

vllm-mlx prend en charge le traitement audio via [mlx-audio](https://github.com/Blaizzy/mlx-audio), offrant :

- **STT (Speech-to-Text)** : Whisper, Parakeet
- **TTS (Text-to-Speech)** : Kokoro, Chatterbox, VibeVoice, VoxCPM
- **Traitement audio** : SAM-Audio (séparation vocale)

## Installation

```bash
# Support audio de base
pip install mlx-audio>=0.2.9

# Required dependencies for TTS
pip install sounddevice soundfile scipy numba tiktoken misaki spacy num2words loguru phonemizer

# Download spacy English model
python -m spacy download en_core_web_sm

# For non-English TTS (Spanish, French, etc.), install espeak-ng:
# macOS
brew install espeak-ng

# Ubuntu/Debian
# sudo apt-get install espeak-ng
```

Ou installez toutes les dépendances audio en une seule commande :

```bash
pip install vllm-mlx[audio]
python -m spacy download en_core_web_sm
brew install espeak-ng  # macOS, for non-English languages
```

## Démarrage rapide

### Speech-to-Text (Transcription)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Transcribe audio file
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=f,
        language="en"  # optional
    )
print(transcript.text)
```

### Text-to-Speech (Génération)

```python
# Generate speech
audio = client.audio.speech.create(
    model="kokoro",
    input="Hello, how are you?",
    voice="af_heart",
    speed=1.0
)

# Save to file
with open("output.wav", "wb") as f:
    f.write(audio.content)
```

### Séparation vocale (SAM-Audio)

Isolez une voix du bruit de fond, de la musique ou d'autres sons :

```python
from vllm_mlx.audio import AudioProcessor

# Load SAM-Audio model
processor = AudioProcessor("mlx-community/sam-audio-large-fp16")
processor.load()

# Separate speech from audio
result = processor.separate("meeting_with_music.mp3", description="speech")

# Save isolated voice and background
processor.save(result.target, "voice_only.wav")
processor.save(result.residual, "background_only.wav")
```

**Exemple en ligne de commande :**
```bash
python examples/audio_separation_example.py meeting.mp3 --play
python examples/audio_separation_example.py song.mp3 --description music -o music.wav
```

### Démo de séparation de batterie

Isolez la batterie d'une chanson rock avec SAM-Audio :

| Audio | Description | Écouter |
|-------|-------------|---------|
| Original | "Get Ready" de David Fesliyan (30s, libre de droits) | [rock_get_ready.mp3](../../../examples/rock_get_ready.mp3) |
| Batterie isolée | Batterie extraite par SAM-Audio | [drums_isolated.wav](../../../examples/drums_isolated.wav) |
| Sans batterie | Piste sans batterie | [rock_no_drums.wav](../../../examples/rock_no_drums.wav) |

```bash
# Isolate drums from rock song
python examples/audio_separation_example.py examples/rock_get_ready.mp3 \
  --description "drums" \
  --output drums_isolated.wav \
  --background rock_no_drums.wav
```

**Performance :** 30 secondes d'audio traitées en environ 20 secondes sur M4 Max.

## Modèles pris en charge

### Modèles STT (Speech-to-Text)

| Modèle | Alias | Langues | Vitesse | Qualité |
|--------|-------|---------|---------|---------|
| `mlx-community/whisper-large-v3-mlx` | `whisper-large-v3` | 99+ | Moyenne | Meilleure |
| `mlx-community/whisper-large-v3-turbo` | `whisper-large-v3-turbo` | 99+ | Rapide | Excellente |
| `mlx-community/whisper-medium-mlx` | `whisper-medium` | 99+ | Rapide | Bonne |
| `mlx-community/whisper-small-mlx` | `whisper-small` | 99+ | Très rapide | Correcte |
| `mlx-community/parakeet-tdt-0.6b-v2` | `parakeet` | Anglais | La plus rapide | Excellente |
| `mlx-community/parakeet-tdt-0.6b-v3` | `parakeet-v3` | Anglais | La plus rapide | Meilleure |

**Recommandations :**
- Multilingue : `whisper-large-v3`
- Anglais uniquement : `parakeet` (3 fois plus rapide)

### Modèles TTS (Text-to-Speech)

#### Kokoro (Rapide, Léger) - Recommandé

| Modèle | Alias | Taille | Langues |
|--------|-------|--------|---------|
| `mlx-community/Kokoro-82M-bf16` | `kokoro` | 82M | EN, ES, FR, JA, ZH, HI, IT, PT |
| `mlx-community/Kokoro-82M-4bit` | `kokoro-4bit` | 82M | EN, ES, FR, JA, ZH, HI, IT, PT |

**Voix (11) :**
- Femme américaine : `af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`
- Homme américain : `am_adam`, `am_michael`
- Femme britannique : `bf_emma`, `bf_isabella`
- Homme britannique : `bm_george`, `bm_lewis`

**Codes de langue :**
| Code | Langue | Code | Langue |
|------|--------|------|--------|
| `a` / `en` | English (US) | `e` / `es` | Español |
| `b` / `en-gb` | English (UK) | `f` / `fr` | Français |
| `j` / `ja` | 日本語 | `z` / `zh` | 中文 |
| `i` / `it` | Italiano | `p` / `pt` | Português |
| `h` / `hi` | हिन्दी | | |

#### Chatterbox (Multilingue, Expressif)

| Modèle | Alias | Taille | Langues |
|--------|-------|--------|---------|
| `mlx-community/chatterbox-turbo-fp16` | `chatterbox` | 134M | 15+ langues |
| `mlx-community/chatterbox-turbo-4bit` | `chatterbox-4bit` | 134M | 15+ langues |

**Langues prises en charge :** EN, ES, FR, DE, IT, PT, RU, JA, ZH, KO, AR, HI, NL, PL, TR

#### VibeVoice (Temps réel)

| Modèle | Alias | Taille | Cas d'usage |
|--------|-------|--------|-------------|
| `mlx-community/VibeVoice-Realtime-0.5B-4bit` | `vibevoice` | 200M | Faible latence, anglais |

#### VoxCPM (Chinois/Anglais)

| Modèle | Alias | Taille | Langues |
|--------|-------|--------|---------|
| `mlx-community/VoxCPM1.5` | `voxcpm` | 0.9B | ZH, EN |
| `mlx-community/VoxCPM1.5-4bit` | `voxcpm-4bit` | 200M | ZH, EN |

### Modèles de traitement audio

#### SAM-Audio (Séparation vocale)

| Modèle | Taille | Cas d'usage |
|--------|--------|-------------|
| `mlx-community/sam-audio-large-fp16` | 3B | Meilleure qualité |
| `mlx-community/sam-audio-large` | 3B | Standard |
| `mlx-community/sam-audio-small-fp16` | 0.6B | Rapide |
| `mlx-community/sam-audio-small` | 0.6B | Léger |

## Référence API

### POST /v1/audio/transcriptions

Transcrit un fichier audio en texte (compatible API OpenAI Whisper).

**Paramètres :**
- `file` : Fichier audio (mp3, wav, m4a, webm)
- `model` : Nom ou alias du modèle
- `language` : Code de langue (optionnel, détection automatique)
- `response_format` : `json` ou `text`

**Limites :**
- Taille maximale par défaut : 25 MiB
- Modifiable avec `--max-audio-upload-mb`

**Exemple :**
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F model=whisper-large-v3
```

### POST /v1/audio/speech

Génère de la parole à partir de texte (compatible API OpenAI TTS).

**Paramètres :**
- `model` : Nom ou alias du modèle
- `input` : Texte à synthétiser
- `voice` : Identifiant de la voix
- `speed` : Vitesse de parole (0,5 à 2,0)
- `response_format` : `wav`, `mp3`

**Limites :**
- Nombre de caractères maximal par défaut : 4096
- Modifiable avec `--max-tts-input-chars`

**Exemple :**
```bash
curl http://localhost:8000/v1/audio/speech \
  -d '{"model": "kokoro", "input": "Hello world", "voice": "af_heart"}' \
  -H "Content-Type: application/json" \
  --output speech.wav
```

### GET /v1/audio/voices

Liste les voix disponibles pour un modèle.

**Exemple :**
```bash
curl http://localhost:8000/v1/audio/voices?model=kokoro
```

## Exemples en ligne de commande

### Transcription en direct / Sous-titres en temps réel

Transcription STT en temps réel depuis votre microphone :

```bash
# Closed captions with whisper-large-v3 (best quality)
python examples/closed_captions.py --language es --chunk 5

# Faster model for lower latency
python examples/closed_captions.py --language en --model whisper-turbo --chunk 3

# Basic mic transcription (record then transcribe)
python examples/mic_transcribe.py --language es

# Real-time chunked transcription
python examples/mic_realtime.py --language es --chunk 3

# Live transcription with voice activity detection
python examples/mic_live.py --language es
```

**Prérequis :**
```bash
pip install sounddevice soundfile numpy
```

### TTS de base

```bash
# Simple TTS example
python examples/tts_example.py "Hello, how are you?" --play

# With different voice
python examples/tts_example.py "Hello!" --voice am_michael --play

# Save to file
python examples/tts_example.py "Welcome to the demo" -o greeting.wav

# List available voices
python examples/tts_example.py --list-voices
```

### TTS multilingue

```bash
# English (auto-selects best model)
python examples/tts_multilingual.py "Hello world" --play

# Spanish
python examples/tts_multilingual.py "Hola mundo" --lang es --play

# French
python examples/tts_multilingual.py "Bonjour le monde" --lang fr --play

# Japanese
python examples/tts_multilingual.py "こんにちは" --lang ja --play

# Chinese
python examples/tts_multilingual.py "你好世界" --lang zh --play

# Use specific model
python examples/tts_multilingual.py "Hello" --model chatterbox --play

# List all models
python examples/tts_multilingual.py --list-models

# List all languages
python examples/tts_multilingual.py --list-languages
```

### Exemples d'assistant vocal professionnel

Exemples vocaux prégénérés avec des **voix natives** pour des cas d'usage professionnels courants :

| Langue | Voix | Message | Écouter |
|--------|------|---------|---------|
| Anglais | af_heart | "Welcome to First National Bank. How may I assist you today?" | [assistant_bank_en.wav](../../../examples/assistant_bank_en.wav) |
| Espagnol | ef_dora | "Gracias por llamar a servicio al cliente. Un agente le atenderá pronto." | [assistant_service_es.wav](../../../examples/assistant_service_es.wav) |
| Français | ff_siwis | "Bienvenue. Votre appel est important pour nous." | [assistant_callcenter_fr.wav](../../../examples/assistant_callcenter_fr.wav) |
| Chinois | zf_xiaobei | "欢迎致电技术支持中心。我们将竭诚为您服务。" | [assistant_support_zh.wav](../../../examples/assistant_support_zh.wav) |

**Générez vos propres exemples avec des voix natives :**
```bash
# English - Bank assistant (native voice: af_heart)
python -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 \
  --text "Welcome to First National Bank. How may I assist you today?" \
  --voice af_heart --lang_code a --file_prefix assistant_bank_en

# Spanish - Customer service (native voice: ef_dora)
python -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 \
  --text "Gracias por llamar a servicio al cliente. Un agente le atendera pronto." \
  --voice ef_dora --lang_code e --file_prefix assistant_service_es

# French - Call center (native voice: ff_siwis)
python -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 \
  --text "Bienvenue. Votre appel est important pour nous." \
  --voice ff_siwis --lang_code f --file_prefix assistant_callcenter_fr

# Chinese - Tech support (native voice: zf_xiaobei)
python -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 \
  --text "欢迎致电技术支持中心。我们将竭诚为您服务。" \
  --voice zf_xiaobei --lang_code z --file_prefix assistant_support_zh
```

### Référence des voix natives

| Langue | Code | Voix |
|--------|------|------|
| English (US) | `a` | af_heart, af_bella, af_nicole, am_adam, am_michael |
| English (UK) | `b` | bf_emma, bf_isabella, bm_george, bm_lewis |
| Espagnol | `e` | ef_dora, em_alex, em_santa |
| Français | `f` | ff_siwis |
| Chinois | `z` | zf_xiaobei, zf_xiaoni, zf_xiaoxiao, zm_yunjian, zm_yunxi |
| Japonais | `j` | jf_alpha, jf_gongitsune, jm_kumo |
| Italien | `i` | if_sara, im_nicola |
| Portugais | `p` | pf_dora, pm_alex |
| Hindi | `h` | hf_alpha, hf_beta, hm_omega |

## API Python

### Utilisation directe (sans serveur)

```python
from vllm_mlx.audio import STTEngine, TTSEngine, AudioProcessor

# Speech-to-Text
stt = STTEngine("mlx-community/whisper-large-v3-mlx")
stt.load()
result = stt.transcribe("audio.mp3")
print(result.text)

# Text-to-Speech
tts = TTSEngine("mlx-community/Kokoro-82M-bf16")
tts.load()
audio = tts.generate("Hello world", voice="af_heart")
tts.save(audio, "output.wav")

# Voice Separation
processor = AudioProcessor("mlx-community/sam-audio-large-fp16")
processor.load()
result = processor.separate("mixed_audio.mp3", description="speech")
processor.save(result.target, "voice_only.wav")
processor.save(result.residual, "background.wav")
```

### Fonctions utilitaires

```python
from vllm_mlx.audio import transcribe_audio, generate_speech, separate_voice

# Quick transcription
result = transcribe_audio("audio.mp3")
print(result.text)

# Quick TTS
audio = generate_speech("Hello world", voice="af_heart")

# Quick voice separation
voice, background = separate_voice("mixed.mp3")
```

## Audio dans le chat

Incluez de l'audio dans les messages du chat (transcrit automatiquement) :

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Summarize this audio"},
            {"type": "audio_url", "audio_url": {"url": "file://meeting.mp3"}}
        ]
    }]
)
```

## Benchmarks

Testé sur Apple M2 Max (32 Go).

### Benchmarks TTS (Kokoro-82M-bf16)

| Longueur du texte | Durée audio | Temps de génération | RTF | Caractères/s |
|-------------------|-------------|---------------------|-----|--------------|
| 25 caractères | 1,95 s | 0,43 s | 4,6x | 58,5 |
| 88 caractères | 6,00 s | 0,32 s | 18,6x | 272,4 |
| 117 caractères | 7,92 s | 0,27 s | 29,0x | 427,4 |

**Résumé :**
- Temps de chargement du modèle : environ 1,0 s
- RTF moyen : **17,4x** (17 fois plus rapide que le temps réel)
- Caractères/s moyens : **252,8**

### Benchmarks STT

| Modèle | Temps de chargement | Transcription (6 s audio) | RTF |
|--------|---------------------|---------------------------|-----|
| whisper-small | 0,25 s | 0,20 s | 30,2x |
| whisper-medium | 18,1 s | 0,38 s | 15,5x |
| whisper-large-v3 | environ 30 s | environ 0,6 s | environ 10x |
| parakeet | environ 0,5 s | environ 0,15 s | environ 40x |

**Notes :**
- Le RTF (Real-Time Factor) indique combien de fois plus rapide que le temps réel
- Le premier chargement inclut le téléchargement du modèle depuis HuggingFace
- Les chargements suivants utilisent les modèles mis en cache

### Recommandations par cas d'usage

| Cas d'usage | Modèle recommandé | Pourquoi |
|-------------|------------------|----------|
| STT anglais rapide | `parakeet` | RTF 40x, faible consommation mémoire |
| STT multilingue | `whisper-large-v3` | 99+ langues |
| STT faible latence | `whisper-small` | RTF 30x, chargement rapide |
| TTS général | `kokoro` | RTF 17x, bonne qualité |
| TTS faible mémoire | `kokoro-4bit` | Quantification 4 bits |

## Conseils de performance

1. **Utilisez Parakeet pour l'anglais** : 40 fois plus rapide que le temps réel
2. **Utilisez les modèles 4 bits** pour réduire la consommation mémoire
3. **Utilisez SAM-Audio small** pour une séparation vocale plus rapide
4. **Mettez les modèles en cache** : les moteurs sont chargés à la demande et mis en cache
5. **Pré-téléchargez les modèles** pour éviter la latence au premier démarrage

## Dépannage

### mlx-audio non installé
```
pip install mlx-audio>=0.2.9
```

### Téléchargement du modèle lent
Les modèles sont téléchargés depuis HuggingFace lors de la première utilisation. Utilisez `huggingface-cli download` pour les pré-télécharger :
```bash
huggingface-cli download mlx-community/whisper-large-v3-mlx
huggingface-cli download mlx-community/Kokoro-82M-bf16
```

### Mémoire insuffisante
Utilisez des modèles plus petits ou des versions quantifiées en 4 bits :
- `whisper-small-mlx` plutôt que `whisper-large-v3-mlx`
- `Kokoro-82M-4bit` plutôt que `Kokoro-82M-bf16`
- `sam-audio-small` plutôt que `sam-audio-large`

### Bug multilingue Kokoro (mlx-audio 0.2.9)

Si vous obtenez `ValueError: too many values to unpack` en utilisant des langues autres que l'anglais (espagnol, chinois, japonais, etc.) avec Kokoro, appliquez ce correctif :

```python
# Fix for mlx_audio/tts/models/kokoro/pipeline.py line 443
# Change:
#     ps, _ = self.g2p(chunk)
# To:
g2p_result = self.g2p(chunk)
ps = g2p_result[0] if isinstance(g2p_result, tuple) else g2p_result
```

**Correctif en une ligne :**
```bash
python -c "
import os
path = os.path.join(os.path.dirname(__import__('mlx_audio').__file__), 'tts/models/kokoro/pipeline.py')
with open(path, 'r') as f: content = f.read()
old = '                    ps, _ = self.g2p(chunk)'
new = '''                    # Fix: handle both tuple (en) and string (zh/ja/es) returns from g2p
                    g2p_result = self.g2p(chunk)
                    ps = g2p_result[0] if isinstance(g2p_result, tuple) else g2p_result'''
if old in content:
    with open(path, 'w') as f: f.write(content.replace(old, new))
    print('Fix applied!')
"
```

Ce bug survient car le g2p anglais retourne un tuple `(phonemes, tokens)` tandis que les autres langues retournent uniquement une chaîne de caractères.
