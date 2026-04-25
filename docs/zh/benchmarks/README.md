# 基准测试

vllm-mlx 在 Apple Silicon 上的性能基准测试。

## 基准测试类型

- [LLM 基准测试](llm.md) - 文本生成性能
- [图像基准测试](image.md) - 图像理解性能
- [视频基准测试](video.md) - 视频理解性能

## 常用命令

```bash
# LLM benchmark
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit

# Image benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Video benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --video
```

## 独立测试默认值

独立基准测试脚本内置了默认模型，可以直接运行：

```bash
python tests/test_continuous_batching.py
python tests/test_prefix_cache.py
```

默认模型：
- `tests/test_continuous_batching.py` 对应 `mlx-community/Qwen3-8B-6bit`
- `tests/test_prefix_cache.py` 对应 `mlx-community/Qwen3-0.6B-8bit`

如需测试其他模型，使用可选的 `--model` 参数：

```bash
python tests/test_continuous_batching.py --model mlx-community/Qwen3-0.6B-8bit
python tests/test_prefix_cache.py --model mlx-community/Qwen3-8B-6bit
```

## 硬件配置

以下 Apple Silicon 配置已收录基准测试结果：

| 芯片 | 内存 | Python |
|------|--------|--------|
| Apple M4 Max | 128 GB unified | 3.13 |
| Apple M1 Max | 64 GB unified | 3.12 |

不同 Apple Silicon 芯片的测试结果会有所差异。

## 贡献基准测试

如果您使用的是其他 Apple Silicon 芯片，欢迎分享您的测试结果：

```bash
vllm-mlx-bench --model mlx-community/Qwen3-0.6B-8bit --output results.json
```

请在 [GitHub Issues](https://github.com/waybarrios/vllm-mlx/issues) 中提交您的结果。
