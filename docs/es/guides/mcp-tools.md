# MCP y Tool Calling

vllm-mlx soporta el Model Context Protocol (MCP) para integrar herramientas externas con LLMs.

## Como funciona el tool calling

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

## Inicio rápido

### 1. Crear la configuración de MCP

Crea el archivo `mcp.json`:

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

### 2. Iniciar el servidor con MCP

```bash
# Modo simple
vllm-mlx serve mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

# Continuous batching
vllm-mlx serve mlx-community/Qwen3-4B-4bit --mcp-config mcp.json --continuous-batching
```

### 3. Verificar el estado de MCP

```bash
# Verificar estado de MCP
curl http://localhost:8000/v1/mcp/status

# Listar las herramientas disponibles
curl http://localhost:8000/v1/mcp/tools
```

## Ejemplo de tool calling

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

## Endpoints de MCP

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/v1/mcp/status` | GET | Verificar el estado de MCP |
| `/v1/mcp/tools` | GET | Listar las herramientas disponibles |
| `/v1/mcp/execute` | POST | Ejecutar una herramienta |

## Ejemplos de servidores MCP

### Sistema de archivos

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

## Multiples servidores MCP

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

## Chat MCP interactivo

Para probar MCP de forma interactiva:

```bash
python examples/mcp_chat.py
```

## Formatos de herramientas soportados

vllm-mlx soporta 12 tool call parsers que cubren todas las familias de modelos principales. Consulta [Tool Calling](tool-calling.md) para ver la lista completa de parsers, alias y ejemplos.

## Seguridad

vllm-mlx incluye medidas de seguridad para prevenir ataques de inyeccion de comandos a traves de servidores MCP.

### Lista blanca de comandos

Solo se permiten comandos de confianza por defecto:

| Categoria | Comandos permitidos |
|----------|-----------------|
| Node.js | `npx`, `npm`, `node` |
| Python | `uvx`, `uv`, `python`, `python3`, `pip`, `pipx` |
| Docker | `docker` |
| Servidores MCP | `mcp-server-*` (servidores oficiales) |

### Patrones bloqueados

Los siguientes patrones estan bloqueados para prevenir ataques de inyeccion:

- Encadenamiento de comandos: `;`, `&&`, `||`, `|`
- Sustitucion de comandos: `` ` ``, `$()`
- Recorrido de rutas: `../`
- Variables de entorno peligrosas: `LD_PRELOAD`, `PATH`, `PYTHONPATH`

### Ejemplo: ataque bloqueado

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

Esta configuración sera rechazada:
```
ValueError: MCP server 'malicious': Command 'bash' is not in the allowed commands whitelist.
```

### Modo de desarrollo (inseguro)

Solo para desarrollo, es posible omitir la validación de seguridad:

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

**ADVERTENCIA**: nunca uses `skip_security_validation` en produccion.

### Lista blanca personalizada

Para agregar comandos personalizados a la lista blanca mediante código:

```python
from vllm_mlx.mcp import MCPCommandValidator, set_validator

# Add custom commands
validator = MCPCommandValidator(
    custom_whitelist={"my-trusted-server", "another-server"}
)
set_validator(validator)
```

## Sandboxing de ejecución de herramientas

Ademas de la validación de comandos, vllm-mlx ofrece sandboxing en tiempo de ejecución para las herramientas:

### Caracteristicas del sandbox

| Caracteristica | Descripcion |
|---------|-------------|
| Lista blanca de herramientas | Permite ejecutar solo herramientas especificas |
| Lista negra de herramientas | Bloquea herramientas peligrosas especificas |
| Validacion de argumentos | Bloquea patrones peligrosos en los argumentos de las herramientas |
| Limite de frecuencia | Limita las llamadas a herramientas por minuto |
| Registro de auditoria | Registra todas las ejecuciones de herramientas |

### Patrones de argumentos bloqueados

Los argumentos de las herramientas son validados para detectar patrones peligrosos:

- Recorrido de rutas: `../`
- Directorios del sistema: `/etc/`, `/proc/`, `/sys/`
- Acceso root: `/root/`, `~root`

### Deteccion de herramientas de alto riesgo

Las herramientas que coincidan con estos patrones generan advertencias de seguridad:

- `execute`, `run_command`, `shell`, `eval`, `exec`, `system`, `subprocess`

### Configuracion personalizada del sandbox

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

### Acceso a los registros de auditoria

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

### Redaccion de datos sensibles

Los registros de auditoria redactan automaticamente los campos sensibles (password, token, secret, key, credential, auth) y truncan los valores de gran tamano.

## Solucion de problemas

### El servidor MCP no se conecta

Verifica que el comando del servidor MCP sea correcto:
```bash
npx -y @modelcontextprotocol/server-filesystem /tmp
```

### La herramienta no se ejecuta

Verifica que la herramienta este disponible:
```bash
curl http://localhost:8000/v1/mcp/tools | jq '.tools[].name'
```

### La llamada a la herramienta no se analiza

Asegurate de usar un modelo que soporte llamadas a funciones (Qwen3, Llama-3.2-Instruct).

### El comando no esta en la lista blanca

Si ves "Command X is not in the allowed commands whitelist", puedes:
1. Usar un comando permitido (ver lista blanca arriba)
2. Agregar el comando a una lista blanca personalizada
3. Usar `skip_security_validation: true` (solo para desarrollo)
