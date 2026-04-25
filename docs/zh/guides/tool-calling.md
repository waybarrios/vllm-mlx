# Tool Calling

vllm-mlx 支持与 OpenAI 兼容的 tool calling（function calling），并为多种主流模型系列提供自动解析。

## 快速开始

启动服务器时添加 `--enable-auto-tool-choice` 标志即可启用 tool calling：

```bash
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

然后通过标准 OpenAI API 使用工具：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }]
)

# Check for tool calls
if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"Function: {tc.function.name}")
        print(f"Arguments: {tc.function.arguments}")
```

## 支持的 tool parser

使用 `--tool-call-parser` 为您的模型系列选择对应的 tool parser：

| Parser | 别名 | 模型 | 格式 |
|--------|------|------|------|
| `auto` | | 任意模型 | 自动检测格式（依次尝试所有 parser） |
| `mistral` | | Mistral、Devstral | `[TOOL_CALLS]` JSON 数组 |
| `qwen` | `qwen3` | Qwen、Qwen3 | `<tool_call>` XML 或 `[Calling tool:]` |
| `llama` | `llama3`、`llama4` | Llama 3.x、4.x | `<function=name>` 标签 |
| `hermes` | `nous` | Hermes、NousResearch | `<tool_call>` XML 包裹的 JSON |
| `deepseek` | `deepseek_v3`、`deepseek_r1` | DeepSeek V3、R1 | Unicode 分隔符 |
| `kimi` | `kimi_k2`、`moonshot` | Kimi K2、Moonshot | `<\|tool_call_begin\|>` 标记 |
| `granite` | `granite3` | IBM Granite 3.x、4.x | `<\|tool_call\|>` 或 `<tool_call>` |
| `nemotron` | `nemotron3` | NVIDIA Nemotron | `<tool_call><function=...><parameter=...>` |
| `xlam` | | Salesforce xLAM | 含 `tool_calls` 数组的 JSON |
| `functionary` | `meetkai` | MeetKai Functionary | 多个 function 块 |
| `glm47` | `glm4` | GLM-4.7、GLM-4.7-Flash | `<tool_call>` 配合 `<arg_key>`/`<arg_value>` XML |

## 模型示例

### Mistral / Devstral

```bash
# Devstral Small（针对编程和 tool use 优化）
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral

# Mistral Instruct
vllm-mlx serve mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral
```

### Qwen

```bash
# Qwen3
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --enable-auto-tool-choice --tool-call-parser qwen
```

### Llama

```bash
# Llama 3.2
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit \
  --enable-auto-tool-choice --tool-call-parser llama
```

### DeepSeek

```bash
# DeepSeek V3
vllm-mlx serve mlx-community/DeepSeek-V3-0324-4bit \
  --enable-auto-tool-choice --tool-call-parser deepseek
```

### IBM Granite

```bash
# Granite 4.0
vllm-mlx serve mlx-community/granite-4.0-tiny-preview-4bit \
  --enable-auto-tool-choice --tool-call-parser granite
```

### NVIDIA Nemotron

```bash
# Nemotron 3 Nano
vllm-mlx serve mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-6Bit \
  --enable-auto-tool-choice --tool-call-parser nemotron
```

### GLM-4.7

```bash
# GLM-4.7 Flash
vllm-mlx serve lmstudio-community/GLM-4.7-Flash-MLX-8bit \
  --enable-auto-tool-choice --tool-call-parser glm47
```

### Kimi K2

```bash
# Kimi K2
vllm-mlx serve mlx-community/Kimi-K2-Instruct-4bit \
  --enable-auto-tool-choice --tool-call-parser kimi
```

### Salesforce xLAM

```bash
# xLAM
vllm-mlx serve mlx-community/xLAM-2-fc-r-4bit \
  --enable-auto-tool-choice --tool-call-parser xlam
```

## Auto Parser

如果不确定使用哪个 tool parser，`auto` parser 会尝试自动检测格式：

```bash
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --enable-auto-tool-choice --tool-call-parser auto
```

auto parser 按以下顺序依次尝试各种格式：

1. Mistral（`[TOOL_CALLS]`）
2. Qwen 括号格式（`[Calling tool:]`）
3. Nemotron（`<tool_call><function=...><parameter=...>`）
4. Qwen/Hermes XML（`<tool_call>{...}</tool_call>`）
5. Llama（`<function=name>{...}</function>`）
6. 原始 JSON

## Streaming Tool Calls

Tool calling 支持 streaming。模型生成完毕后发送 tool call 信息：

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's 25 * 17?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate math expressions",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.tool_calls:
        for tc in chunk.choices[0].delta.tool_calls:
            print(f"Tool call: {tc.function.name}({tc.function.arguments})")
```

## 处理工具返回结果

收到 tool call 后，执行对应函数并将结果返回给模型：

```python
import json

# 第一次请求，模型决定调用工具
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[weather_tool]
)

# 获取 tool call
tool_call = response.choices[0].message.tool_calls[0]
tool_call_id = tool_call.id
function_name = tool_call.function.name
arguments = json.loads(tool_call.function.arguments)

# 执行函数（由您自行实现）
result = get_weather(**arguments)  # {"temperature": 22, "condition": "sunny"}

# 将结果返回给模型
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {"role": "assistant", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)}
    ],
    tools=[weather_tool]
)

print(response.choices[0].message.content)
# "The weather in Tokyo is sunny with a temperature of 22C."
```

## Think 标签处理

会产生 `<think>...</think>` reasoning 标签的模型（如 DeepSeek-R1、Qwen3、GLM-4.7）均可自动处理。tool parser 会在提取 tool call 前剥离 thinking 内容，因此 reasoning 标签不会干扰 tool call 解析。

即使 `<think>` 是通过提示词注入的（即仅有闭合标签 `</think>` 的隐式 think 标签），也同样适用。

## CLI 参数参考

| 选项 | 说明 |
|------|------|
| `--enable-auto-tool-choice` | 启用自动 tool calling |
| `--tool-call-parser` | 选择 tool parser（见上表） |

完整选项请参阅 [CLI Reference](../reference/cli.md)。
