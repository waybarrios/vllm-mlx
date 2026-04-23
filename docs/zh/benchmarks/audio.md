# 音频基准测试

## 语音转文本 (STT) 基准测试

### 运行 STT 基准测试

```bash
# Run with default test audio
python examples/benchmark_audio.py --stt

# Run with your own audio file
python examples/benchmark_audio.py --stt --audio path/to/audio.wav
```

### 测试结果（M4 Max，128GB）

**测试音频：** 46.7 秒的合成语音

| Model | Parameters | Load Time | Transcribe Time | RTF* |
|-------|------------|-----------|-----------------|------|
| whisper-tiny | 39M | 0.34s | 0.24s | **197x** |
| whisper-small | 244M | 0.18s | 0.47s | **98x** |
| whisper-medium | 769M | 0.35s | 1.15s | **41x** |
| whisper-large-v3 | 1.5B | 0.50s | 1.96s | **24x** |
| whisper-large-v3-turbo | 809M | 0.12s | 0.86s | **55x** |

*RTF = 实时倍率（值越高速度越快）。RTF 为 100x 表示 1 分钟的音频约在 0.6 秒内转录完成。*

### 测试结果（M1 Max，64GB）

使用 Parakeet 进行 STT（默认环境，Whisper 因 numpy 依赖版本不匹配而不可用）：

| Model | Load Time | Transcribe Time | RTF |
|-------|-----------|-----------------|-----|
| parakeet-tdt-0.6b-v2 | 0.28s | 1.01s | **9.9x** |
| parakeet-tdt-0.6b-v3 | 0.30s | 0.19s | **52.7x** |

使用 Whisper 进行 STT（显式指定 `numpy==2.3.5` 并搭配 `uv run --no-sync`）：

| Model | Load Time | Transcribe Time | RTF |
|-------|-----------|-----------------|-----|
| whisper-tiny | 4.02s | 1.05s | **9.5x** |
| whisper-small | 10.15s | 1.03s | **9.7x** |
| whisper-medium | 22.96s | 2.20s | **4.6x** |
| whisper-large-v3 | 38.34s | 0.96s | **10.5x** |
| whisper-large-v3-turbo | 21.79s | 0.70s | **14.3x** |
| parakeet-tdt-0.6b-v2 | 0.47s | 0.18s | **54.4x** |
| parakeet-tdt-0.6b-v3 | 1.13s | 0.18s | **54.6x** |

### 模型推荐

| 使用场景 | 推荐模型 | 原因 |
|----------|----------|------|
| **实时转录** | whisper-tiny | 速度最快（197x RTF），延迟低 |
| **通用场景** | whisper-large-v3-turbo | 速度（55x）与质量兼顾，综合表现最佳 |
| **最高精度** | whisper-large-v3 | 准确率最高，支持 99 种以上语言 |
| **低内存** | whisper-small | 244M 参数，质量良好 |

### 转录质量

所有模型均能正确转录测试音频。示例输出：

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

### 支持的语言

Whisper 模型支持 99 种以上语言，包括：
- 英语、西班牙语、法语、德语、意大利语、葡萄牙语
- 中文（普通话、粤语）、日语、韩语
- 阿拉伯语、印地语、俄语、土耳其语、乌克兰语
- 以及更多语言

## 文本转语音 (TTS) 基准测试

### 运行 TTS 基准测试

```bash
python examples/benchmark_audio.py --tts
```

### 测试结果（M4 Max，128GB）

**测试内容：** 为 3 段文本样本（短、中、长）生成音频

| Model | Load Time | Chars/sec | RTF* |
|-------|-----------|-----------|------|
| Kokoro-82M-bf16 | 0.8s | 350+ | **22x** |
| Kokoro-82M-4bit | 0.4s | 320+ | **20x** |

*RTF = 实时倍率。RTF 为 22x 表示 1 秒的音频约在 0.045 秒内生成完毕。*

### TTS 测试结果（M1 Max，64GB）

| Model | Load Time | Avg Chars/s | Avg RTF |
|-------|-----------|-------------|---------|
| Kokoro-82M-bf16 | 2.81s | 176.0 | **11.9x** |
| Kokoro-82M-4bit | 0.22s | 225.6 | **15.5x** |

### TTS 质量

Kokoro 可生成自然流畅的语音，具备以下特性：
- 11 种内置音色（男声与女声）
- 支持 8 种语言（英语、西班牙语、法语、日语、中文、意大利语、葡萄牙语、印地语）
- 82M 参数，轻量且高效

## 音频处理基准测试

### SAM-Audio（音源分离）

**测试内容：** 从 30 秒摇滚歌曲中分离鼓声

| Metric | Value |
|--------|-------|
| Model | sam-audio-large-fp16 |
| Processing time | ~20s |
| Peak memory | ~27 GB |
| Output sample rate | 48000 Hz |

## 运行全部音频基准测试

```bash
# Run all benchmarks
python examples/benchmark_audio.py --all

# Or run individually
python examples/benchmark_audio.py --stt
python examples/benchmark_audio.py --tts
```

## mlx-community 上的可用模型

### STT 模型
- `mlx-community/whisper-tiny-mlx`
- `mlx-community/whisper-small-mlx`
- `mlx-community/whisper-medium-mlx`
- `mlx-community/whisper-large-v3-mlx`
- `mlx-community/whisper-large-v3-turbo`
- `mlx-community/parakeet-tdt-0.6b-v2`
- `mlx-community/parakeet-tdt-0.6b-v3`

### TTS 模型
- `mlx-community/Kokoro-82M-bf16`（推荐）
- `mlx-community/Kokoro-82M-4bit`
- `mlx-community/chatterbox-turbo-fp16`
- `mlx-community/VibeVoice-Realtime-0.5B-4bit`

### 音频处理
- `mlx-community/sam-audio-large-fp16`
