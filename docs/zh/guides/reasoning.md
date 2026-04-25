# Reasoning 模型

vllm-mlx 支持在给出答案之前展示 thinking 过程的 reasoning 模型。Qwen3 和 DeepSeek-R1 等模型会将 reasoning 内容包裹在 `<think>...</think>` 标签中，vllm-mlx 可以解析这些标签，将 reasoning 与最终回答分离。

## 为什么使用 Reasoning 解析？

reasoning 模型生成的原始输出通常如下所示：

```
<think>
Let me analyze this step by step.
First, I need to consider the constraints.
The answer should be a prime number less than 10.
Checking: 2, 3, 5, 7 are all prime and less than 10.
</think>
The prime numbers less than 10 are: 2, 3, 5, 7.
```

不启用 reasoning 解析时，响应中会包含原始标签。启用 reasoning parsing 后，thinking 过程与最终回答会被分离到 API 响应的不同字段中。

## 快速开始

### 启动服务器并指定 Reasoning Parser

```bash
# For Qwen3 models
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# For DeepSeek-R1 models
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

### API 响应格式

启用 reasoning parsing 后，API 响应中会包含 `reasoning` 字段。

**非 streaming 响应：**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The prime numbers less than 10 are: 2, 3, 5, 7.",
      "reasoning": "Let me analyze this step by step.\nFirst, I need to consider the constraints.\nThe answer should be a prime number less than 10.\nChecking: 2, 3, 5, 7 are all prime and less than 10."
    }
  }]
}
```

**Streaming 响应：**

reasoning 和正文内容分块独立发送。在 reasoning 阶段，数据块的 `reasoning` 字段有内容；当模型进入最终回答阶段后，数据块的 `content` 字段有内容：

```json
{"delta": {"reasoning": "Let me analyze"}}
{"delta": {"reasoning": " this step by step."}}
{"delta": {"reasoning": "\nFirst, I need to"}}
...
{"delta": {"content": "The prime"}}
{"delta": {"content": " numbers less than 10"}}
{"delta": {"content": " are: 2, 3, 5, 7."}}
```

## 与 OpenAI SDK 配合使用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Non-streaming
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What are the prime numbers less than 10?"}]
)

message = response.choices[0].message
print("Reasoning:", message.reasoning)  # The thinking process
print("Answer:", message.content)        # The final answer
```

### Streaming 与 Reasoning

```python
reasoning_text = ""
content_text = ""

stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Solve: 2 + 2 = ?"}],
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if hasattr(delta, 'reasoning') and delta.reasoning:
        reasoning_text += delta.reasoning
        print(f"[Thinking] {delta.reasoning}", end="")
    if delta.content:
        content_text += delta.content
        print(delta.content, end="")

print(f"\n\nFinal reasoning: {reasoning_text}")
print(f"Final answer: {content_text}")
```

## 支持的 Parser

### Qwen3 Parser（`qwen3`）

适用于使用显式 `<think>` 和 `</think>` 标签的 Qwen3 模型。

- 需要开标签和闭标签**同时存在**
- 如果标签缺失，输出将被视为普通内容
- 适合：Qwen3-0.6B、Qwen3-4B、Qwen3-8B 及同系列模型

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

### DeepSeek-R1 Parser（`deepseek_r1`）

适用于可能省略开标签 `<think>` 的 DeepSeek-R1 模型。

- 比 Qwen3 parser 更宽松
- 能处理 `<think>` 为隐式的情况
- 即使没有 `<think>`，`</think>` 之前的内容也会被视为 reasoning

```bash
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

## 工作原理

reasoning parser 通过基于文本的检测来识别模型输出中的 thinking 标签。在 streaming 过程中，它会追踪当前在输出中的位置，将每个 token 正确路由到 `reasoning` 或 `content` 字段。

```
Model Output:        <think>Step 1: analyze...</think>The answer is 42.
                     ├─────────────────────┤├─────────────────────┤
Parsed:              │     reasoning       ││       content       │
                     └─────────────────────┘└─────────────────────┘
```

解析过程是无状态的，通过累积文本来判断上下文，在 token 以任意分块到达的 streaming 场景下也能稳定工作。

## 最佳使用建议

### 提示词写法

引导模型逐步思考，reasoning 模型的效果更好：

```python
messages = [
    {"role": "system", "content": "Think through problems step by step before answering."},
    {"role": "user", "content": "What is 17 × 23?"}
]
```

### 处理缺失的 Reasoning

某些提示词可能不会触发 reasoning。此时 `reasoning` 值为 `None`，所有输出都进入 `content`：

```python
message = response.choices[0].message
if message.reasoning:
    print(f"Model's thought process: {message.reasoning}")
print(f"Answer: {message.content}")
```

### 温度参数与 Reasoning

较低的温度通常会产生更稳定的 reasoning 模式：

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    temperature=0.3  # More focused reasoning
)
```

## 向后兼容性

未指定 `--reasoning-parser` 时，服务器行为与之前一致：thinking 标签包含在 `content` 字段中，响应中不会添加 `reasoning` 字段。这确保现有应用无需修改即可继续正常使用。

## 示例：数学题求解器

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

def solve_math(problem: str) -> dict:
    """Solve a math problem and return reasoning + answer."""
    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a math tutor. Show your work."},
            {"role": "user", "content": problem}
        ],
        temperature=0.2
    )

    message = response.choices[0].message
    return {
        "problem": problem,
        "work": message.reasoning,
        "answer": message.content
    }

result = solve_math("If a train travels 120 km in 2 hours, what is its average speed?")
print(f"Problem: {result['problem']}")
print(f"\nWork shown:\n{result['work']}")
print(f"\nFinal answer: {result['answer']}")
```

## Curl 示例

### 非 Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 15% of 80?"}]
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 15% of 80?"}],
    "stream": true
  }'
```

## 常见问题排查

### 响应中没有 reasoning 字段

- 确认启动服务器时指定了 `--reasoning-parser`
- 检查模型是否实际使用了 thinking 标签（并非所有提示词都会触发 reasoning）

### Reasoning 出现在 content 中

- 模型可能没有使用预期的标签格式
- 尝试换用其他 parser（`qwen3` 或 `deepseek_r1`）

### Reasoning 被截断

- 如果模型在 thinking 过程中触及了 token 上限，请增大 `--max-tokens`

## 相关链接

- [支持的模型](../reference/models.md). 支持 reasoning 的模型列表
- [服务器配置](server.md). 所有服务器选项
- [CLI 参考](../reference/cli.md). 命令行选项
