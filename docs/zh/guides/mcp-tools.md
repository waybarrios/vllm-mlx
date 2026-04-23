# MCP 与 tool calling

vllm-mlx 支持 Model Context Protocol (MCP)，用于将外部工具与 LLM 集成。

## tool calling 工作原理

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Tool Calling Flow                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. User Request                                                    │
│     ─────────────────►  "List files in /tmp"                       │
│                                                                     │
│  2. LLM Generates Tool Call                                         │
│     ─────────────────►  tool_calls: [{                             │
│                           name: "list_directory",                   │
│                           arguments: {path: "/tmp"}                 │
│                         }]                                          │
│                                                                     │
│  3. App Executes Tool via MCP                                       │
│     ─────────────────►  MCP Server executes list_directory         │
│                         Returns: ["file1.txt", "file2.txt"]        │
│                                                                     │
│  4. Tool Result Sent Back to LLM                                    │
│     ─────────────────►  role: "tool", content: [...]               │
│                                                                     │
│  5. LLM Generates Final Response                                    │
│     ─────────────────►  "The /tmp directory contains 2 files..."   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 创建 MCP 配置

创建 `mcp.json`：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

### 2. 启动带 MCP 的服务器

```bash
# 简单模式
vllm-mlx serve mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

# 连续批处理
vllm-mlx serve mlx-community/Qwen3-4B-4bit --mcp-config mcp.json --continuous-batching
```

### 3. 验证 MCP 状态

```bash
# 查看 MCP 状态
curl http://localhost:8000/v1/mcp/status

# 列出可用工具
curl http://localhost:8000/v1/mcp/tools
```

## tool calling 示例

```python
import json
import httpx

BASE_URL = "http://localhost:8000"

# 1. Get available tools
tools_response = httpx.get(f"{BASE_URL}/v1/mcp/tools")
tools = tools_response.json()["tools"]

# 2. Send request with tools
response = httpx.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "default",
        "messages": [{"role": "user", "content": "List files in /tmp"}],
        "tools": tools,
        "max_tokens": 1024
    }
)

result = response.json()
message = result["choices"][0]["message"]

# 3. Check for tool calls
if message.get("tool_calls"):
    tool_call = message["tool_calls"][0]

    # 4. Execute tool via MCP
    exec_response = httpx.post(
        f"{BASE_URL}/v1/mcp/execute",
        json={
            "server": "filesystem",
            "tool": tool_call["function"]["name"],
            "arguments": json.loads(tool_call["function"]["arguments"])
        }
    )
    tool_result = exec_response.json()

    # 5. Send result back to LLM
    messages = [
        {"role": "user", "content": "List files in /tmp"},
        message,
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps(tool_result["result"])
        }
    ]

    final_response = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={"model": "default", "messages": messages}
    )
    print(final_response.json()["choices"][0]["message"]["content"])
```

## MCP 接口端点

| 端点 | 方法 | 说明 |
|----------|--------|-------------|
| `/v1/mcp/status` | GET | 查看 MCP 状态 |
| `/v1/mcp/tools` | GET | 列出可用工具 |
| `/v1/mcp/execute` | POST | 执行工具 |

## MCP 服务器示例

### 文件系统

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

### GitHub

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token"
      }
    }
  }
}
```

### PostgreSQL

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost/db"
      }
    }
  }
}
```

### Brave Search

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-key"
      }
    }
  }
}
```

## 使用多个 MCP 服务器

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token"
      }
    }
  }
}
```

## 交互式 MCP 聊天

如需交互式测试 MCP：

```bash
python examples/mcp_chat.py
```

## 支持的工具格式

vllm-mlx 支持 12 种 tool call parser，覆盖所有主流模型系列。完整的 parser 列表、别名及示例请参见 [Tool Calling](tool-calling.md)。

## 安全性

vllm-mlx 内置安全措施，防止通过 MCP 服务器进行命令注入攻击。

### 命令白名单

默认情况下，仅允许可信命令：

| 类别 | 允许的命令 |
|----------|-----------------|
| Node.js | `npx`、`npm`、`node` |
| Python | `uvx`、`uv`、`python`、`python3`、`pip`、`pipx` |
| Docker | `docker` |
| MCP 服务器 | `mcp-server-*`（官方服务器） |

### 屏蔽模式

以下模式会被屏蔽以防止注入攻击：

- 命令链接：`;`、`&&`、`||`、`|`
- 命令替换：`` ` ``、`$()`
- 路径穿越：`../`
- 危险环境变量：`LD_PRELOAD`、`PATH`、`PYTHONPATH`

### 示例：被屏蔽的攻击

```json
{
  "mcpServers": {
    "malicious": {
      "command": "bash",
      "args": ["-c", "rm -rf /"]
    }
  }
}
```

此配置将被拒绝：
```
ValueError: MCP server 'malicious': Command 'bash' is not in the allowed commands whitelist.
```

### 开发模式（不安全）

仅限开发环境，可绕过安全校验：

```json
{
  "mcpServers": {
    "custom": {
      "command": "my-custom-server",
      "skip_security_validation": true
    }
  }
}
```

**警告**：切勿在生产环境中使用 `skip_security_validation`。

### 自定义白名单

如需通过编程方式向白名单添加自定义命令：

```python
from vllm_mlx.mcp import MCPCommandValidator, set_validator

# Add custom commands
validator = MCPCommandValidator(
    custom_whitelist={"my-trusted-server", "another-server"}
)
set_validator(validator)
```

## 工具执行沙箱

除命令校验外，vllm-mlx 还为工具执行提供运行时沙箱。

### 沙箱功能

| 功能 | 说明 |
|---------|-------------|
| 工具白名单 | 仅允许特定工具执行 |
| 工具黑名单 | 屏蔽特定危险工具 |
| 参数校验 | 屏蔽工具参数中的危险模式 |
| 频率限制 | 限制每分钟的工具调用次数 |
| 审计日志 | 记录所有工具执行情况 |

### 屏蔽的参数模式

工具参数会针对以下危险模式进行校验：

- 路径穿越：`../`
- 系统目录：`/etc/`、`/proc/`、`/sys/`
- root 访问：`/root/`、`~root`

### 高风险工具检测

匹配以下模式的工具会触发安全警告：

- `execute`、`run_command`、`shell`、`eval`、`exec`、`system`、`subprocess`

### 自定义沙箱配置

```python
from vllm_mlx.mcp import ToolSandbox, set_sandbox

# Create sandbox with custom settings
sandbox = ToolSandbox(
    # Only allow specific tools (whitelist mode)
    allowed_tools={"read_file", "list_directory"},

    # Block specific tools (blacklist mode)
    blocked_tools={"execute_command", "run_shell"},

    # Rate limit: max 30 calls per minute
    max_calls_per_minute=30,

    # Optional audit callback
    audit_callback=lambda audit: print(f"Tool: {audit.tool_name}, Success: {audit.success}"),
)
set_sandbox(sandbox)
```

### 访问审计日志

```python
from vllm_mlx.mcp import get_sandbox

sandbox = get_sandbox()

# Get recent audit entries
entries = sandbox.get_audit_log(limit=50)

# Filter by tool name
file_ops = sandbox.get_audit_log(tool_filter="file")

# Get only errors
errors = sandbox.get_audit_log(errors_only=True)

# Clear audit log
sandbox.clear_audit_log()
```

### 敏感数据脱敏

审计日志会自动对敏感字段（password、token、secret、key、credential、auth）进行脱敏处理，并对过大的值进行截断。

## 故障排查

### MCP 服务器无法连接

检查 MCP 服务器命令是否正确：
```bash
npx -y @modelcontextprotocol/server-filesystem /tmp
```

### 工具无法执行

验证工具是否可用：
```bash
curl http://localhost:8000/v1/mcp/tools | jq '.tools[].name'
```

### tool call 未被解析

请确保所用模型支持函数调用（如 Qwen3、Llama-3.2-Instruct）。

### 命令不在白名单中

如果看到 "Command X is not in the allowed commands whitelist"，可采取以下措施之一：
1. 使用允许的命令（参见上方白名单）
2. 将该命令添加到自定义白名单
3. 使用 `skip_security_validation: true`（仅限开发环境）
