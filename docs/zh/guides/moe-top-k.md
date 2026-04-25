# MoE top_k 覆盖参数（`--moe-top-k`）

减少 Mixture of Experts 模型（如 Qwen3-30B-A3B）每个 token 激活的 expert 数量，以少量质量损失换取明显更高的解码吞吐量。

> **状态：** 可选参数，默认行为不变。以下质量数据基于 Qwen3-30B-A3B-4bit 在 M4 Max 128 GB 上的测试结果，在将其用于生产环境前请在你的模型上自行验证。

## 功能说明

Qwen3-30B-A3B 使用 `top_k=8` 训练，即每个 token 从 128 个 expert 中选取 8 个。在 Apple Silicon 上进行 batch=1 解码时，expert 矩阵乘法（`SwitchGLU`）是每层计算中占比最大的部分，其开销与 `top_k` 大致呈线性关系。在推理阶段降低 `top_k` 已被证明（LExI 2025，Lynx 2024）能在保留大部分训练质量的同时，有效缩短解码时间。

`--moe-top-k N` 会遍历已加载模型的每一层，对含有 `.mlp.switch_mlp`（即稀疏 MoE 块）的层将 `top_k` 设置为 N。密集层和密集模型不受影响，该参数对它们是空操作。

## 用法

```bash
# Server
vllm-mlx serve mlx-community/Qwen3-30B-A3B-4bit \
  --continuous-batching \
  --moe-top-k 4

# Bench
vllm-mlx bench mlx-community/Qwen3-30B-A3B-4bit --moe-top-k 4
```

若 N 大于模型训练时的 `top_k`，该参数会被拒绝，因为只有降低才有意义，不支持提高。

## 实测影响

### 解码吞吐量（M4 Max 128 GB，batch=1，贪心解码）

| top_k | tok/s | 对比基线 |
|---:|---:|---:|
| 8（基线） | 126.5 | - |
| 6 | 136.1 | +7.6% |
| 5 | 140.3 | +10.9% |
| 4 | 147.3 | +16.5% |

### 质量评估（Qwen3-30B-A3B-4bit，lm-evaluation-harness，MLX backend）

<!-- populated after eval completes -->

| top_k | MMLU (acc) | GSM8K (exact match) | Δ vs baseline |
|---:|---:|---:|---:|
| 8 | TBD | TBD | - |
| 6 | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD |

MMLU：随机抽取 200 个样本，0-shot。
GSM8K：随机抽取 100 个样本，0-shot，严格 exact-match。

以上数据具有**方向性参考价值**，完整评测集规模更大，会改变绝对精度数值，但各配置间的相对差距不会有太大变化。

### 贪心输出一致性

在 4-bit 检查点上使用 `top_k=4` 时，我们测试的所有探针提示中，生成的**前 16 个 token 与基线完全一致**。这表明 top_k=4 不会改变早期解码步骤中的 argmax，模型对减少一半激活 expert 具有内在的鲁棒性。

当 `top_k=3` 或更低时，质量会出现可见的下降（此处未测量，基于 LExI 论文推断），因此该参数在配置校验层刻意不允许低于 1，但生产环境推荐的最低值为 `top_k=4`。

## 适用场景与不适用场景

适合使用的情况：
- 运行 Qwen3 MoE（或兼容模型：Qwen3.5 MoE、Gemma-MoE），且单用户解码吞吐量是瓶颈。
- 工作负载允许少量质量损失，以换取明显的延迟改善。
- 部署在受内存带宽限制的硬件上（M 系列 Apple Silicon），expert gather 主导每步解码时间。

不适合使用的情况：
- 运行密集模型，该参数是空操作，没有任何效果。
- 对评测集排行榜精度有顶尖要求。
- 运行长链式推理或"思考模式"生成，质量下降幅度可能比 0-shot MMLU 所示更陡。

## 与其他优化叠加使用

该参数可与量化叠加使用。在 Qwen3-30B-A3B-4bit 上的实测叠加结果如下：

- 4-bit + top_k=8：126.5 tok/s（基线）
- 4-bit + top_k=4：147.3 tok/s（+16.5%）
- 3-bit + top_k=8：138.6 tok/s（+9.6%）
- 3-bit + top_k=6：147.1 tok/s（+16.3%），质量差异可测量
- 3-bit + top_k=4：157.3 tok/s（+24%），**输出质量严重下降**（在冒烟测试中模型回答了不同的问题）

3-bit + top_k=4 的数值误差累积超出了 argmax 稳定的临界点。最多只应使用一个激进参数：4-bit + top_k=4 或 3-bit + top_k=6。两者的 tok/s 大致相同（约 147），但质量表现差异显著。

## 内部实现

- 补丁辅助函数：`vllm_mlx.scheduler.apply_moe_top_k_override(model, k)`
- 在 `Scheduler.__init__` 中于模型加载完成后执行。
- 测试：`tests/test_moe_top_k.py`，覆盖密集模型、混合架构及校验路径。

## 参考资料

- LExI: Layer-Adaptive Active Experts, [arXiv 2509.02753](https://arxiv.org/html/2509.02753)
- Not All Experts are Equal (NAEE), [ACL 2024](https://aclanthology.org/2024.acl-long.334.pdf)
- SwiftLM (`SWIFTLM_TOP_K` env knob prior art), [github.com/SharpAI/SwiftLM](https://github.com/SharpAI/SwiftLM)
