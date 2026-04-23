# Warm Prompts

在服务器启动时预先填充 prefix cache，使 agent 发送的**第一个**请求命中已预热的缓存，而无需为其数千字节的系统提示支付完整的 prefill 开销。

## 适用场景

Agent 工作负载，如代理编码助手或推理助手的代理、MCP 服务器、多 agent 编排器，始终会发送相同的系统提示。在当前实现中，冷启动服务器收到的第一个请求需要为该系统提示支付完整的 prefill 代价。对于数十亿参数的模型，这意味着数秒的 TTFT，而此时用户正在等待其新 agent 首次响应。

如果您在部署时已知道各 agent 的系统提示，可将其写入一个 JSON 文件并通过 `--warm-prompts` 指向它。服务器会在启动时对每条提示执行一次 `max_tokens=1` 的聊天补全，KV cache 状态随即落入 prefix cache，后续真实请求即可通过严格前缀匹配命中缓存。

此功能需要 `--continuous-batching`（prefix cache 依赖该模式）。

## 快速示例

```bash
# 一次性写入您关心的 agent
cat > ~/.config/vllm-mlx/agents.json <<'JSON'
[
  [{"role": "system", "content": "You are a code assistant..."}]
]
JSON

# 将服务器指向该文件
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --continuous-batching \
  --warm-prompts ~/.config/vllm-mlx/agents.json
```

启动时您将看到：

```
[lifespan] Warm-up done (strict-prefix): 1 completed, 0 skipped,
           1431 prompt tokens in 0.2s
```

第一个共享已预热系统提示的真实请求将命中缓存，其 `tokens_saved` 接近预热提示的长度。

## 文件格式

顶层为一个 JSON 列表，每个条目本身也是一个聊天消息列表，结构与 `/v1/chat/completions` 中的 `messages` 字段相同。

```json
[
  [
    {"role": "system", "content": "You are a code assistant..."}
  ],
  [
    {"role": "system", "content": "You are a senior code reviewer..."}
  ],
  [
    {"role": "system", "content": "You are a planner..."},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "Hello, what are we planning?"}
  ]
]
```

单条系统提示是最常见的用法。多轮历史也受支持，适用于需要预热特定对话开头的场景（少样本示例、持续运行的助手角色等）。

## 规模建议

预热提示通过 `asyncio.gather` **并发**处理，因此 N 条条目会在启动时触发 N 个并发 prefill，每个 prefill 会为其提示长度分配 KV cache。

**建议条目数为 1 至 3 条**，足以覆盖典型 agent 部署的热路径（每个角色一条）。在内存紧张的模型上，过大的 warm-prompts 文件可能在启动时耗尽可用空间。

如需预热数十个角色，请提交一个 issue 并说明您的工作负载，我们可以添加 `--warm-prompts-concurrency=N` 上限参数。

## 基准测试

**测试环境：** M4 Max，128 GB 统一内存。每次测量使用两个独立服务器（冷启动与预热），隔离冷启动。`long` 提示集（约 2500 个用户 token）前置约 1700 token 的系统提示以匹配预热提示。`max_tokens=128`。bench-serve 使用 `--skip-preflight-token-count`，避免 count_prompt_tokens 预检污染缓存。

| 模型 | 并发 | 冷启动 TTFT | 预热 TTFT | 加速比 |
|------|-----:|----------:|----------:|------:|
| Qwen3-0.6B-8bit | 1 | 563 ms | 419 ms | 1.34x |
| Qwen3-0.6B-8bit | 4 | 1 723 ms | 1 282 ms | 1.34x |
| Qwen3-0.6B-8bit | 8 | 3 708 ms | 2 661 ms | 1.39x |
| Llama-3.2-3B-Instruct-4bit | 1 | 1 754 ms | 1 060 ms | 1.65x |
| Llama-3.2-3B-Instruct-4bit | 4 | 5 926 ms | 3 945 ms | 1.50x |
| Llama-3.2-3B-Instruct-4bit | 8 | 15 161 ms | 9 820 ms | 1.54x |
| Qwen3-4B-4bit | 1 | 4 937 ms | 2 191 ms | 2.25x |
| Qwen3-4B-4bit | 4 | 12 535 ms | 9 623 ms | 1.30x |
| Qwen3-4B-4bit | 8 | 38 148 ms | 23 878 ms | 1.60x |
| Qwen3.6-35B-A3B-4bit (MoE/hybrid) | 1 | 2 400 ms | 1 603 ms | 1.50x |
| Qwen3.6-35B-A3B-4bit | 4 | 8 735 ms | 6 054 ms | 1.44x |
| Qwen3.6-35B-A3B-4bit | 8 | 22 419 ms | 14 409 ms | 1.56x |

全部 12 项配置均有提升。当提示占总长度比例最高时（并发=1，长系统提示），TTFT 节省最为显著，在并发负载下仍有实质性收益。

**生成 tok/s** 对于稠密模型基本持平（误差在 ±5% 以内）。Qwen3.6-35B-A3B（MoE）在并发数大于等于 4 时出现 20 至 35% 的解码速度下降，原因似乎是 MoE 路由与批量调度之间的交互。对于 agent 工作负载，TTFT 节省仍主导端到端延迟，但若您的工作流在高并发下以解码为瓶颈，请注意这一点。

## 工作原理

朴素的预热方式，即用占位用户消息渲染聊天模板并缓存 token，对于混合 SSM+attention 模型（Qwen3.5-MoE、Qwen3.6-MoE）不适用。这类模型的缓存层包含无法裁剪的 SSM 状态，因此 `memory_cache.py` 禁用了 LCP 匹配。占位用户内容与真实用户内容不同，基于 token 的缓存条目不再是任何真实请求的严格前缀。

本预热器会用两个不同的用户内容（`"__PROBE_A__"` 和 `"__PROBE_B__"`）**两次**渲染聊天模板，找到两个字符串开始发散的字符位置，并在该边界处截断第一次渲染的结果。这段截断后的字符串，即用户内容被插入之前的全部内容，是发送给引擎的内容。

由于引擎的真实请求路径同样使用 `tokenize=False` 渲染模板，再由分词器对结果进行编码，因此预热生成的 token 保证是任何具有匹配系统提示且聊天历史为空的真实请求的严格前缀。严格前缀匹配适用于所有缓存层类型，包括禁用 LCP 的混合路径。

## 管理操作

### 清除内存中的 prefix cache

```bash
curl -X DELETE http://localhost:8000/v1/cache/prefix
```

若服务器以 `--warm-prompts` 启动，清除后会在后台重新执行预热。响应会立即返回，不等待重新预热完成。

响应：

```json
{"status": "cleared", "rewarm_scheduled": true}
```

### 查看缓存状态

```bash
curl http://localhost:8000/v1/status | jq '.cache'
```

使用 warm-prompts 启动后，在第一个用户请求到来之前，您将看到 `entry_count > 0`。

## 针对您自己的场景进行基准测试

如需测量对您的模型和提示的实际影响，请使用 `bench-serve`：

```bash
# 冷启动：不使用 warm-prompts
vllm-mlx serve MODEL --continuous-batching &
vllm-mlx bench-serve --prompts long --concurrency 1,4 \
  --system-prompt-file my-system.txt --tag cold \
  --output cold.csv --format csv

# 预热：相同服务器配置 + --warm-prompts
vllm-mlx serve MODEL --continuous-batching \
  --warm-prompts ~/.config/vllm-mlx/agents.json &
vllm-mlx bench-serve --prompts long --concurrency 1,4 \
  --system-prompt-file my-system.txt --tag warm \
  --output warm.csv --format csv
```

设置 `--system-prompt-file` 时会自动启用 `--skip-preflight-token-count`，防止 `count_prompt_tokens` 预检污染缓存。比较 `cold.csv` 与 `warm.csv` 即可评估您工作负载的实际效果。
