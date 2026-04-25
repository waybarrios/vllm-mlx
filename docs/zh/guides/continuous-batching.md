# Continuous Batching

Continuous batching 在同时服务多个用户时能显著提升 throughput。

## 启用 Continuous Batching

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --continuous-batching
```

## 与 Paged Cache 配合使用

启用高效内存的前缀共享：

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit --continuous-batching --use-paged-cache
```

## 工作原理

### 简单模式（默认）
- 每次处理一个请求
- 单用户场景下 throughput 最高
- 无 batching 额外开销

### Continuous Batching 模式
- 多个请求同时处理
- 并发用户场景下 throughput 更高
- 每个请求存在少量额外开销

### Paged Cache
- KV cache 以固定大小的块存储
- 相同的系统提示词共享同一批块
- 10 个以上并发用户时节省 80% 以上内存

## 性能测试结果

**Continuous Batching 测试结果（M4 Max，128GB）：**

| 模型 | 单请求 | Batch（5 个请求） | 加速比 |
|-------|----------------|---------------|---------|
| Llama-3.2-1B-Instruct-4bit | 299.1 tok/s | 613.0 tok/s | **2.05x** |
| Llama-3.2-3B-Instruct-4bit | 137.6 tok/s | 208.1 tok/s | **1.51x** |
| Qwen3-0.6B-8bit | 328.1 tok/s | 1111.8 tok/s | **3.39x** |
| Qwen3-30B-A3B-4bit | 98.1 tok/s | 233.3 tok/s | **2.38x** |
| Qwen2.5-1.5B-Instruct-4bit | 196.9 tok/s | 322.2 tok/s | **1.64x** |

*5 个并发请求的 batching 可将 throughput 提升 1.5 到 3 倍。*

## Streaming 性能

**Streaming 性能（M4 Max，128GB）：**

| 模型 | TTFT | 生成速度 |
|-------|------|------------------|
| Llama-3.2-1B-Instruct-4bit | ~4.6ms | 218.9 tok/s |
| Llama-3.2-3B-Instruct-4bit | ~10.7ms | 93.6 tok/s |
| Qwen3-0.6B-8bit | ~3.0ms | 328.5 tok/s |
| Qwen3-30B-A3B-4bit | ~10.2ms | 98.4 tok/s |
| Qwen2.5-1.5B-Instruct-4bit | ~7.1ms | 140.3 tok/s |

*TTFT = Time to First Token（首 token 延迟）*

## Streaming 配置

使用 `--stream-interval` 控制 token 发送频率：

```bash
# 每个 token 立即发送（最流畅）
vllm-mlx serve model --continuous-batching --stream-interval 1

# 批量发送 token（适合高延迟场景）
vllm-mlx serve model --continuous-batching --stream-interval 5
```

| 值 | 行为 |
|-------|----------|
| `1` | 每个 token 立即发送 |
| `2-5` | 攒批后再发送 |
| `10+` | 最大化 throughput，输出颗粒度更大 |

## 内存管理

对于大型模型，prefix cache 可能占用大量内存。内存感知缓存会自动进行管理：

```bash
# 自动检测（使用可用内存的 20%）
vllm-mlx serve model --continuous-batching

# 显式限制
vllm-mlx serve model --continuous-batching --cache-memory-mb 2048

# 自定义百分比
vllm-mlx serve model --continuous-batching --cache-memory-percent 0.10
```

| 选项 | 说明 |
|--------|-------------|
| `--cache-memory-mb` | 以 MB 为单位设置显式上限 |
| `--cache-memory-percent` | 可用内存的占比（默认值：0.20） |
| `--no-memory-aware-cache` | 使用基于条目数量的旧式缓存 |

## Prefix Cache

Prefix caching 对重复提示词复用 KV cache。

### 工作原理

```
User 1: System prompt (500 tokens) → Creates 8 blocks
User 2: Same system prompt → Shares 8 blocks (ref_count++)
User N: Same system prompt → Shares 8 blocks (ref_count++)

Memory savings: 80%+ for 10+ concurrent users
```

### 缓存键策略

- **LLM**：`hash(prompt)`
- **图片**：`hash(image_content) + hash(prompt)`
- **视频**：`hash(video_path) + hash(fps) + hash(max_frames) + hash(prompt)`

### 测试 Prefix Cache

```bash
python tests/test_prefix_cache.py
```

```
======================================================================
  LLM PREFIX CACHE TEST
======================================================================
  Model: mlx-community/Qwen3-0.6B-8bit
  Expected behavior:
    - Same prompt → cache HIT
    - Different prompt → cache MISS or PREFIX_HIT (shared template tokens)
----------------------------------------------------------------------
  Results:
  Step   | Description         | Expected | Actual | Status
  -------+---------------------+----------+--------+-------
  1a     | First request       | MISS     | MISS   | PASS
  1b     | Same prompt         | HIT      | HIT    | PASS
  1c     | Different prompt    | MISS     | MISS   | PASS
  1d     | Return to prompt 1  | HIT      | HIT    | PASS
======================================================================
```

## 运行基准测试

```bash
# Continuous batching 基准测试
python tests/test_continuous_batching.py

# Prefix cache 测试
python tests/test_prefix_cache.py
```

## 适用场景

| 场景 | 模式 |
|----------|------|
| 单用户，追求最高速度 | 简单模式（默认） |
| 多用户并发 | `--continuous-batching` |
| 大型模型（7B+） | `--continuous-batching --cache-memory-mb 2048` |
| 生产环境，提示词共享 | `--continuous-batching --use-paged-cache` |

## 生产环境配置

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --port 8000
```
