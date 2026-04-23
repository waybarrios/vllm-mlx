# MCP & Tool Calling

vllm-mlx prend en charge le Model Context Protocol (MCP) pour intégrer des outils externes avec des LLM.

## Fonctionnement du tool calling

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

## Démarrage rapide

### 1. Créer la configuration MCP

Créez `mcp.json` :

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

### 2. Démarrer le serveur avec MCP

```bash
# Mode simple
vllm-mlx serve mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

# Continuous batching
vllm-mlx serve mlx-community/Qwen3-4B-4bit --mcp-config mcp.json --continuous-batching
```

### 3. Vérifier l'état du MCP

```bash
# Vérifier l'état du MCP
curl http://localhost:8000/v1/mcp/status

# Lister les outils disponibles
curl http://localhost:8000/v1/mcp/tools
```

## Exemple de tool calling

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

## Points de terminaison MCP

| Point de terminaison | Méthode | Description |
|----------------------|---------|-------------|
| `/v1/mcp/status` | GET | Vérifier l'état du MCP |
| `/v1/mcp/tools` | GET | Lister les outils disponibles |
| `/v1/mcp/execute` | POST | Exécuter un outil |

## Exemples de serveurs MCP

### Système de fichiers

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

## Plusieurs serveurs MCP

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

## Chat MCP interactif

Pour tester MCP de manière interactive :

```bash
python examples/mcp_chat.py
```

## Formats d'outils pris en charge

vllm-mlx prend en charge 12 tool call parsers couvrant toutes les grandes familles de modèles. Voir [Tool Calling](tool-calling.md) pour la liste complète des parsers, alias et exemples.

## Sécurité

vllm-mlx inclut des mesures de sécurité pour prévenir les attaques par injection de commandes via les serveurs MCP.

### Liste blanche de commandes

Seules les commandes de confiance sont autorisées par défaut :

| Catégorie | Commandes autorisées |
|-----------|----------------------|
| Node.js | `npx`, `npm`, `node` |
| Python | `uvx`, `uv`, `python`, `python3`, `pip`, `pipx` |
| Docker | `docker` |
| Serveurs MCP | `mcp-server-*` (serveurs officiels) |

### Motifs bloqués

Les motifs suivants sont bloqués pour prévenir les attaques par injection :

- Enchaînement de commandes : `;`, `&&`, `||`, `|`
- Substitution de commandes : `` ` ``, `$()`
- Traversée de répertoires : `../`
- Variables d'environnement dangereuses : `LD_PRELOAD`, `PATH`, `PYTHONPATH`

### Exemple : attaque bloquée

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

Cette configuration sera rejetée :
```
ValueError: MCP server 'malicious': Command 'bash' is not in the allowed commands whitelist.
```

### Mode développement (non sécurisé)

Pour le développement uniquement, il est possible de contourner la validation de sécurité :

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

**AVERTISSEMENT** : N'utilisez jamais `skip_security_validation` en production !

### Liste blanche personnalisée

Pour ajouter des commandes personnalisées à la liste blanche par programmation :

```python
from vllm_mlx.mcp import MCPCommandValidator, set_validator

# Add custom commands
validator = MCPCommandValidator(
    custom_whitelist={"my-trusted-server", "another-server"}
)
set_validator(validator)
```

## Sandboxing de l'exécution des outils

Au-delà de la validation des commandes, vllm-mlx fournit un sandboxing à l'exécution pour les exécutions d'outils.

### Fonctionnalités du sandbox

| Fonctionnalité | Description |
|----------------|-------------|
| Liste blanche d'outils | Autoriser uniquement des outils spécifiques à s'exécuter |
| Liste noire d'outils | Bloquer des outils dangereux spécifiques |
| Validation des arguments | Bloquer les motifs dangereux dans les arguments des outils |
| Limitation du débit | Limiter les appels d'outils par minute |
| Journal d'audit | Suivre toutes les exécutions d'outils |

### Motifs d'arguments bloqués

Les arguments des outils sont validés pour détecter les motifs dangereux :

- Traversée de répertoires : `../`
- Répertoires système : `/etc/`, `/proc/`, `/sys/`
- Accès root : `/root/`, `~root`

### Détection des outils à haut risque

Les outils correspondant à ces motifs déclenchent des avertissements de sécurité :

- `execute`, `run_command`, `shell`, `eval`, `exec`, `system`, `subprocess`

### Configuration personnalisée du sandbox

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

### Accès aux journaux d'audit

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

### Expurgation des données sensibles

Les journaux d'audit expurgent automatiquement les champs sensibles (password, token, secret, key, credential, auth) et tronquent les valeurs volumineuses.

## Dépannage

### Le serveur MCP ne se connecte pas

Vérifiez que la commande du serveur MCP est correcte :
```bash
npx -y @modelcontextprotocol/server-filesystem /tmp
```

### L'outil ne s'exécute pas

Vérifiez que l'outil est disponible :
```bash
curl http://localhost:8000/v1/mcp/tools | jq '.tools[].name'
```

### L'appel d'outil n'est pas analysé

Assurez-vous d'utiliser un modèle qui prend en charge le function calling (Qwen3, Llama-3.2-Instruct).

### La commande n'est pas dans la liste blanche

Si vous voyez « Command X is not in the allowed commands whitelist », vous pouvez :
1. Utiliser une commande autorisée (voir la liste blanche ci-dessus)
2. Ajouter la commande à une liste blanche personnalisée
3. Utiliser `skip_security_validation: true` (développement uniquement)
