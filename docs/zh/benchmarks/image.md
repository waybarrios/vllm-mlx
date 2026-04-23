# 图像基准测试

## 运行图像基准测试

```bash
# 完整基准测试（10 种分辨率）
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# 快速基准测试（4 种分辨率）
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --quick
```

## 测试结果 - Qwen3-VL-8B-Instruct-4bit（M4 Max，128GB）

| Resolution | Pixels | Time | Tokens | Speed |
|------------|--------|------|--------|-------|
| 224x224 | 50K | 1.04s | 78 | 74.8 tok/s |
| 336x336 | 113K | 0.94s | 64 | 68.3 tok/s |
| 448x448 | 201K | 1.45s | 70 | 48.1 tok/s |
| 512x512 | 262K | 1.58s | 99 | 62.8 tok/s |
| 672x672 | 452K | 1.83s | 83 | 45.3 tok/s |
| 768x768 | 590K | 2.05s | 91 | 44.3 tok/s |
| 896x896 | 803K | 2.61s | 90 | 34.5 tok/s |
| 1024x1024 | 1.0M | 2.79s | 76 | 27.2 tok/s |
| 1280x720 | 922K | 2.97s | 96 | 32.4 tok/s |
| 1920x1080 | 2.1M | 6.30s | 89 | 14.1 tok/s |

**摘要：** 所有分辨率的平均速度为 45.2 tok/s。最快为 224x224（74.8 tok/s），最慢为 1920x1080（14.1 tok/s）。

## 测试结果 - Qwen3-VL-8B-Instruct-4bit（M1 Max，64GB）

本地 MLLM 基准测试：

| Resolution | Pixels | Time | Tokens | Speed |
|------------|--------|------|--------|-------|
| 224x224 | 50K | 1.84s | 78 | 42.5 tok/s |
| 448x448 | 201K | 2.28s | 70 | 30.7 tok/s |
| 768x768 | 590K | 4.39s | 91 | 20.7 tok/s |
| 1024x1024 | 1.0M | 6.41s | 76 | 11.9 tok/s |

| Configs | Total Time (s) | Total Tokens | Aggregate Tok/s |
|---------|-----------------|--------------|-----------------|
| 4 | 14.92 | 315 | 21.1 |

## 测试结果 - Qwen3-VL-4B-Instruct-3bit 服务端（M1 Max，64GB）

| Resolution | Pixels | Time | Tokens | Speed |
|------------|--------|------|--------|-------|
| 224x224 | 50K | 1.65s | 113 | 68.4 tok/s |
| 448x448 | 201K | 2.09s | 120 | 57.5 tok/s |
| 768x768 | 590K | 2.93s | 106 | 36.2 tok/s |
| 1024x1024 | 1.0M | 4.12s | 100 | 24.3 tok/s |

| Configs | Total Time (s) | Total Tokens | Aggregate Tok/s |
|---------|-----------------|--------------|-----------------|
| 4 | 10.79 | 439 | 40.7 |

## MLLM 前缀缓存测试结果

```
======================================================================
  MLLM PREFIX CACHE TEST
======================================================================
  Model: mlx-community/Qwen3-VL-4B-Instruct-3bit
  Test: Verify KV cache reuse for repeated image/video + prompt combinations
  Expected behavior:
    - Same image + same prompt → cache HIT
    - Same image + different prompt → cache MISS
    - Different image + same prompt → cache MISS
----------------------------------------------------------------------
  SETUP: Loading Model
----------------------------------------------------------------------
    Model loaded in 0.11s

----------------------------------------------------------------------
  SETUP: Creating Test Images
----------------------------------------------------------------------
    Resized: 224x224, 336x336, 512x512, 768x768

----------------------------------------------------------------------
  TEST 1: Image Cache - Basic Hit/Miss
----------------------------------------------------------------------
    Results:
    Step   | Description               | Expected | Actual | Time   | Status
    -------+---------------------------+----------+--------+--------+-------
    1a     | First image+prompt        | MISS     | MISS   | 0.10ms | ✓
    1b     | Same image+prompt         | HIT      | HIT    | 0.18ms | ✓
    1c     | Different prompt          | MISS     | MISS   | 0.01ms | ✓
    1d     | Return to original        | HIT      | HIT    | 0.18ms | ✓

----------------------------------------------------------------------
  TEST 2: Different Images
----------------------------------------------------------------------
    Results:
    Step   | Description               | Expected | Actual | Time   | Status
    -------+---------------------------+----------+--------+--------+-------
    2a     | Image A first request     | MISS     | MISS   | 0.01ms | ✓
    2b     | Image B first request     | MISS     | MISS   | 0.01ms | ✓
    2c     | Image A cached            | HIT      | HIT    | 0.13ms | ✓

----------------------------------------------------------------------
  TEST 3: Image Resolutions
----------------------------------------------------------------------
    Results:
    Step   | Description           | Expected | Actual | Time   | Status
    -------+-----------------------+----------+--------+--------+-------
    3.1a | 224x224 first         | MISS     | MISS   | 0.01ms | ✓
    3.1b | 224x224 cached        | HIT      | HIT    | 0.20ms | ✓
    3.2a | 336x336 first         | MISS     | MISS   | 0.01ms | ✓
    3.2b | 336x336 cached        | HIT      | HIT    | 0.21ms | ✓
    3.3a | 512x512 first         | MISS     | MISS   | 0.12ms | ✓
    3.3b | 512x512 cached        | HIT      | HIT    | 0.20ms | ✓
    3.4a | 768x768 first         | MISS     | MISS   | 0.12ms | ✓
    3.4b | 768x768 cached        | HIT      | HIT    | 0.24ms | ✓
======================================================================
```

## 缓存键策略

- **图像：** `hash(image_content) + hash(prompt)`

相同图像与相同提示词始终命中缓存。不同图像或不同提示词将不命中缓存。

## 性能提示

- 分辨率越小，处理速度越快（如 224x224 对比 1920x1080）
- 请根据任务需求选择合适的分辨率
- 批量处理尺寸相近的图像，以获得稳定的吞吐量

## 指标说明

| Metric | Description |
|--------|-------------|
| Resolution | 图像尺寸（宽 x 高） |
| Pixels | 总像素数 |
| Time | 生成时间 |
| Tokens | 生成的输出 token 数量 |
| Speed | 每秒 token 数（tok/s） |
