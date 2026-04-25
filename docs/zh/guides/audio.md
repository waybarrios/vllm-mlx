# 音频支持

vllm-mlx 通过 [mlx-audio](https://github.com/Blaizzy/mlx-audio) 支持音频处理，提供以下功能：

- **STT（语音转文字）**：whisper、Parakeet
- **TTS（文字转语音）**：Kokoro、Chatterbox、VibeVoice、VoxCPM
- **音频处理**：SAM-Audio（人声分离）

## 安装

```bash
# 核心音频支持
pip install mlx-audio>=0.2.9

# TTS 所需依赖
pip install sounddevice soundfile scipy numba tiktoken misaki spacy num2words loguru phonemizer

# 下载 spacy 英文模型
python -m spacy download en_core_web_sm

# 非英语 TTS（西班牙语、法语等）需安装 espeak-ng：
# macOS
brew install espeak-ng

# Ubuntu/Debian
# sudo apt-get install espeak-ng
```

或一次性安装所有音频依赖：

```bash
pip install vllm-mlx[audio]
python -m spacy download en_core_web_sm
brew install espeak-ng  # macOS，用于非英语语言
```

## 快速开始

### STT（语音转录）

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 转录音频文件
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=f,
        language="en"  # optional
    )
print(transcript.text)
```

### TTS（语音合成）

```python
# 生成语音
audio = client.audio.speech.create(
    model="kokoro",
    input="Hello, how are you?",
    voice="af_heart",
    speed=1.0
)

# 保存到文件
with open("output.wav", "wb") as f:
    f.write(audio.content)
```

### 人声分离（SAM-Audio）

从背景噪声、音乐或其他声音中提取人声：

```python
from vllm_mlx.audio import AudioProcessor

# 加载 SAM-Audio 模型
processor = AudioProcessor("mlx-community/sam-audio-large-fp16")
processor.load()

# 从音频中分离语音
result = processor.separate("meeting_with_music.mp3", description="speech")

# 保存分离后的人声和背景音
processor.save(result.target, "voice_only.wav")
processor.save(result.residual, "background_only.wav")
```

**命令行示例：**
```bash
python examples/audio_separation_example.py meeting.mp3 --play
python examples/audio_separation_example.py song.mp3 --description music -o music.wav
```

### 鼓声分离演示

使用 SAM-Audio 从摇滚歌曲中分离鼓声：

| 音频 | 说明 | 收听 |
|-------|-------------|--------|
| 原始音频 | David Fesliyan 的"Get Ready"（30 秒，免版权） | [rock_get_ready.mp3](../../../examples/rock_get_ready.mp3) |
| 分离鼓声 | SAM-Audio 提取的鼓声 | [drums_isolated.wav](../../../examples/drums_isolated.wav) |
| 去除鼓声 | 移除鼓声后的音轨 | [rock_no_drums.wav](../../../examples/rock_no_drums.wav) |

```bash
# 从摇滚歌曲中分离鼓声
python examples/audio_separation_example.py examples/rock_get_ready.mp3 \
  --description "drums" \
  --output drums_isolated.wav \
  --background rock_no_drums.wav
```

**性能：** 在 M4 Max 上处理 30 秒音频约需 20 秒。

## 支持的模型

### STT 模型（语音转文字）

| 模型 | 别名 | 语言数 | 速度 | 质量 |
|-------|-------|-----------|-------|---------|
| `mlx-community/whisper-large-v3-mlx` | `whisper-large-v3` | 99+ | 中等 | 最佳 |
| `mlx-community/whisper-large-v3-turbo` | `whisper-large-v3-turbo` | 99+ | 快速 | 优秀 |
| `mlx-community/whisper-medium-mlx` | `whisper-medium` | 99+ | 快速 | 良好 |
| `mlx-community/whisper-small-mlx` | `whisper-small` | 99+ | 极快 | 一般 |
| `mlx-community/parakeet-tdt-0.6b-v2` | `parakeet` | 仅英语 | 最快 | 优秀 |
| `mlx-community/parakeet-tdt-0.6b-v3` | `parakeet-v3` | 仅英语 | 最快 | 最佳 |

**推荐方案：**
- 多语言场景：`whisper-large-v3`
- 仅英语场景：`parakeet`（速度快 3 倍）

### TTS 模型（文字转语音）

#### Kokoro（快速、轻量）- 推荐

| 模型 | 别名 | 参数量 | 支持语言 |
|-------|-------|------|-----------|
| `mlx-community/Kokoro-82M-bf16` | `kokoro` | 82M | EN、ES、FR、JA、ZH、HI、IT、PT |
| `mlx-community/Kokoro-82M-4bit` | `kokoro-4bit` | 82M | EN、ES、FR、JA、ZH、HI、IT、PT |

**声音（11 种）：**
- 美式女声：`af_heart`、`af_bella`、`af_nicole`、`af_sarah`、`af_sky`
- 美式男声：`am_adam`、`am_michael`
- 英式女声：`bf_emma`、`bf_isabella`
- 英式男声：`bm_george`、`bm_lewis`

**语言代码：**
| 代码 | 语言 | 代码 | 语言 |
|------|----------|------|----------|
| `a` / `en` | 英语（美国） | `e` / `es` | Español |
| `b` / `en-gb` | 英语（英国） | `f` / `fr` | Français |
| `j` / `ja` | 日本語 | `z` / `zh` | 中文 |
| `i` / `it` | Italiano | `p` / `pt` | Português |
| `h` / `hi` | हिन्दी | | |

#### Chatterbox（多语言、表现力强）

| 模型 | 别名 | 参数量 | 支持语言 |
|-------|-------|------|-----------|
| `mlx-community/chatterbox-turbo-fp16` | `chatterbox` | 134M | 15+ 种语言 |
| `mlx-community/chatterbox-turbo-4bit` | `chatterbox-4bit` | 134M | 15+ 种语言 |

**支持语言：** EN、ES、FR、DE、IT、PT、RU、JA、ZH、KO、AR、HI、NL、PL、TR

#### VibeVoice（实时）

| 模型 | 别名 | 参数量 | 适用场景 |
|-------|-------|------|----------|
| `mlx-community/VibeVoice-Realtime-0.5B-4bit` | `vibevoice` | 200M | 低延迟、仅英语 |

#### VoxCPM（中英双语）

| 模型 | 别名 | 参数量 | 支持语言 |
|-------|-------|------|-----------|
| `mlx-community/VoxCPM1.5` | `voxcpm` | 0.9B | ZH、EN |
| `mlx-community/VoxCPM1.5-4bit` | `voxcpm-4bit` | 200M | ZH、EN |

### 音频处理模型

#### SAM-Audio（人声分离）

| 模型 | 参数量 | 适用场景 |
|-------|------|----------|
| `mlx-community/sam-audio-large-fp16` | 3B | 最佳质量 |
| `mlx-community/sam-audio-large` | 3B | 标准 |
| `mlx-community/sam-audio-small-fp16` | 0.6B | 快速 |
| `mlx-community/sam-audio-small` | 0.6B | 轻量 |

## API 参考

### POST /v1/audio/transcriptions

将音频转录为文字（兼容 OpenAI Whisper API）。

**参数：**
- `file`：音频文件（mp3、wav、m4a、webm）
- `model`：模型名称或别名
- `language`：语言代码（可选，自动检测）
- `response_format`：`json` 或 `text`

**限制：**
- 默认上传上限：25 MiB
- 可通过 `--max-audio-upload-mb` 修改

**示例：**
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F model=whisper-large-v3
```

### POST /v1/audio/speech

从文字生成语音（兼容 OpenAI TTS API）。

**参数：**
- `model`：模型名称或别名
- `input`：待合成文本
- `voice`：声音 ID
- `speed`：语速（0.5 到 2.0）
- `response_format`：`wav`、`mp3`

**限制：**
- 默认输入上限：4096 个字符
- 可通过 `--max-tts-input-chars` 修改

**示例：**
```bash
curl http://localhost:8000/v1/audio/speech \
  -d '{"model": "kokoro", "input": "Hello world", "voice": "af_heart"}' \
  -H "Content-Type: application/json" \
  --output speech.wav
```

### GET /v1/audio/voices

列出模型可用的声音。

**示例：**
```bash
curl http://localhost:8000/v1/audio/voices?model=kokoro
```

## 命令行示例

### 实时转录与字幕

从麦克风进行实时 STT 转录：

```bash
# 使用 whisper-large-v3 生成字幕（最佳质量）
python examples/closed_captions.py --language es --chunk 5

# 使用更快的模型降低延迟
python examples/closed_captions.py --language en --model whisper-turbo --chunk 3

# 基础麦克风转录（先录音再转录）
python examples/mic_transcribe.py --language es

# 实时分块转录
python examples/mic_realtime.py --language es --chunk 3

# 带语音活动检测的实时转录
python examples/mic_live.py --language es
```

**依赖安装：**
```bash
pip install sounddevice soundfile numpy
```

### 基础 TTS

```bash
# 简单 TTS 示例
python examples/tts_example.py "Hello, how are you?" --play

# 使用不同声音
python examples/tts_example.py "Hello!" --voice am_michael --play

# 保存到文件
python examples/tts_example.py "Welcome to the demo" -o greeting.wav

# 列出可用声音
python examples/tts_example.py --list-voices
```

### 多语言 TTS

```bash
# 英语（自动选择最佳模型）
python examples/tts_multilingual.py "Hello world" --play

# 西班牙语
python examples/tts_multilingual.py "Hola mundo" --lang es --play

# 法语
python examples/tts_multilingual.py "Bonjour le monde" --lang fr --play

# 日语
python examples/tts_multilingual.py "こんにちは" --lang ja --play

# 中文
python examples/tts_multilingual.py "你好世界" --lang zh --play

# 指定模型
python examples/tts_multilingual.py "Hello" --model chatterbox --play

# 列出所有模型
python examples/tts_multilingual.py --list-models

# 列出所有语言
python examples/tts_multilingual.py --list-languages
```

### 商务助手语音示例

使用**原生声音**预生成的常见商务场景语音样本：

| 语言 | 声音 | 内容 | 收听 |
|----------|-------|---------|--------|
| 英语 | af_heart | "Welcome to First National Bank. How may I assist you today?" | [assistant_bank_en.wav](../../../examples/assistant_bank_en.wav) |
| 西班牙语 | ef_dora | "Gracias por llamar a servicio al cliente. Un agente le atenderá pronto." | [assistant_service_es.wav](../../../examples/assistant_service_es.wav) |
| 法语 | ff_siwis | "Bienvenue. Votre appel est important pour nous." | [assistant_callcenter_fr.wav](../../../examples/assistant_callcenter_fr.wav) |
| 中文 | zf_xiaobei | "欢迎致电技术支持中心。我们将竭诚为您服务。" | [assistant_support_zh.wav](../../../examples/assistant_support_zh.wav) |

**使用原生声音自行生成：**
```bash
# 英语 - 银行助手（原生声音：af_heart）
python -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 \
  --text "Welcome to First National Bank. How may I assist you today?" \
  --voice af_heart --lang_code a --file_prefix assistant_bank_en

# 西班牙语 - 客服（原生声音：ef_dora）
python -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 \
  --text "Gracias por llamar a servicio al cliente. Un agente le atendera pronto." \
  --voice ef_dora --lang_code e --file_prefix assistant_service_es

# 法语 - 呼叫中心（原生声音：ff_siwis）
python -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 \
  --text "Bienvenue. Votre appel est important pour nous." \
  --voice ff_siwis --lang_code f --file_prefix assistant_callcenter_fr

# 中文 - 技术支持（原生声音：zf_xiaobei）
python -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 \
  --text "欢迎致电技术支持中心。我们将竭诚为您服务。" \
  --voice zf_xiaobei --lang_code z --file_prefix assistant_support_zh
```

### 原生声音参考

| 语言 | 代码 | 声音 |
|----------|------|--------|
| 英语（美国） | `a` | af_heart、af_bella、af_nicole、am_adam、am_michael |
| 英语（英国） | `b` | bf_emma、bf_isabella、bm_george、bm_lewis |
| 西班牙语 | `e` | ef_dora、em_alex、em_santa |
| 法语 | `f` | ff_siwis |
| 中文 | `z` | zf_xiaobei、zf_xiaoni、zf_xiaoxiao、zm_yunjian、zm_yunxi |
| 日语 | `j` | jf_alpha、jf_gongitsune、jm_kumo |
| 意大利语 | `i` | if_sara、im_nicola |
| 葡萄牙语 | `p` | pf_dora、pm_alex |
| 印地语 | `h` | hf_alpha、hf_beta、hm_omega |

## Python API

### 直接调用（不启动服务器）

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

### 便捷函数

```python
from vllm_mlx.audio import transcribe_audio, generate_speech, separate_voice

# 快速转录
result = transcribe_audio("audio.mp3")
print(result.text)

# 快速 TTS
audio = generate_speech("Hello world", voice="af_heart")

# 快速人声分离
voice, background = separate_voice("mixed.mp3")
```

## 在对话中使用音频

在对话消息中附带音频（自动转录）：

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

## 基准测试

测试环境：Apple M2 Max（32GB）。

### TTS 基准测试（Kokoro-82M-bf16）

| 文本长度 | 音频时长 | 生成耗时 | RTF | 字符/秒 |
|-------------|----------------|----------|-----|-----------|
| 25 字符 | 1.95s | 0.43s | 4.6x | 58.5 |
| 88 字符 | 6.00s | 0.32s | 18.6x | 272.4 |
| 117 字符 | 7.92s | 0.27s | 29.0x | 427.4 |

**汇总：**
- 模型加载时间：约 1.0 秒
- 平均 RTF：**17.4x**（比实时快 17 倍）
- 平均字符/秒：**252.8**

### STT 基准测试

| 模型 | 加载时间 | 转录耗时（6 秒音频） | RTF |
|-------|-----------|----------------------|-----|
| whisper-small | 0.25s | 0.20s | 30.2x |
| whisper-medium | 18.1s | 0.38s | 15.5x |
| whisper-large-v3 | ~30s | ~0.6s | ~10x |
| parakeet | ~0.5s | ~0.15s | ~40x |

**说明：**
- RTF（实时倍率）表示处理速度相对于实时的倍数
- 首次加载包含从 HuggingFace 下载模型的时间
- 后续加载使用已缓存的模型

### 按场景推荐

| 场景 | 推荐模型 | 原因 |
|----------|------------------|-----|
| 英语 STT（快速） | `parakeet` | RTF 40x，内存占用低 |
| 多语言 STT | `whisper-large-v3` | 支持 99+ 种语言 |
| 低延迟 STT | `whisper-small` | RTF 30x，加载快 |
| 通用 TTS | `kokoro` | RTF 17x，质量良好 |
| 低内存 TTS | `kokoro-4bit` | 4-bit 量化 |

## 性能建议

1. **英语场景使用 Parakeet**，比实时快 40 倍
2. **使用 4-bit 模型**降低内存占用
3. **使用 SAM-Audio small**加快人声分离速度
4. **模型缓存**，引擎采用懒加载并自动缓存
5. **提前下载模型**，避免首次运行时的延迟

## 常见问题

### mlx-audio 未安装
```
pip install mlx-audio>=0.2.9
```

### 模型下载缓慢
模型首次使用时从 HuggingFace 下载。可使用 `huggingface-cli download` 提前下载：
```bash
huggingface-cli download mlx-community/whisper-large-v3-mlx
huggingface-cli download mlx-community/Kokoro-82M-bf16
```

### 内存不足
请使用较小的模型或 4-bit 量化版本：
- 用 `whisper-small-mlx` 替代 `whisper-large-v3-mlx`
- 用 `Kokoro-82M-4bit` 替代 `Kokoro-82M-bf16`
- 用 `sam-audio-small` 替代 `sam-audio-large`

### Kokoro 多语言问题（mlx-audio 0.2.9）

使用非英语语言（西班牙语、中文、日语等）时若出现 `ValueError: too many values to unpack`，请应用以下修复：

```python
# Fix for mlx_audio/tts/models/kokoro/pipeline.py line 443
# Change:
#     ps, _ = self.g2p(chunk)
# To:
g2p_result = self.g2p(chunk)
ps = g2p_result[0] if isinstance(g2p_result, tuple) else g2p_result
```

**一键修复：**
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

此问题的原因是英语 g2p 返回元组 `(phonemes, tokens)`，而其他语言仅返回字符串。
