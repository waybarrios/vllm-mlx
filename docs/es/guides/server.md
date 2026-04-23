# Servidor compatible con OpenAI

vllm-mlx provee un servidor FastAPI con compatibilidad completa con la API de OpenAI.

Por defecto, el servidor escucha solo en `127.0.0.1`. Usa `--host 0.0.0.0` solo cuando quieras exponerlo fuera de la máquina local de forma intencional.

## Iniciar el servidor

### Modo simple (por defecto)

Máximo rendimiento para un solo usuario:

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
```

### Modo continuous batching

Para múltiples usuarios concurrentes:

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

### Con paged cache

Caché eficiente en memoria para producción:

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching --use-paged-cache
```

## Opciones del servidor

| Opción | Descripción | Valor por defecto |
|--------|-------------|---------|
| `--port` | Puerto del servidor | 8000 |
| `--host` | Host del servidor | 127.0.0.1 |
| `--api-key` | Clave de API para autenticación | None |
| `--rate-limit` | Solicitudes por minuto por cliente (0 = desactivado) | 0 |
| `--timeout` | Tiempo límite de solicitud en segundos | 300 |
| `--enable-metrics` | Expone métricas de Prometheus en `/metrics` | False |
| `--continuous-batching` | Activa batching para múltiples usuarios | False |
| `--use-paged-cache` | Activa paged KV cache | False |
| `--cache-memory-mb` | Límite de memoria de caché en MB | Auto |
| `--cache-memory-percent` | Fracción de RAM para caché | 0.20 |
| `--max-tokens` | Máximo de tokens por defecto | 32768 |
| `--max-request-tokens` | Máximo de `max_tokens` aceptado de clientes de la API | 32768 |
| `--default-temperature` | Temperatura por defecto cuando no se especifica | None |
| `--default-top-p` | top_p por defecto cuando no se especifica | None |
| `--stream-interval` | Tokens por fragmento de streaming | 1 |
| `--mcp-config` | Ruta al archivo de configuración de MCP | None |
| `--reasoning-parser` | Parser para modelos de reasoning (`qwen3`, `deepseek_r1`) | None |
| `--embedding-model` | Pre-carga un modelo de embeddings al iniciar | None |
| `--enable-auto-tool-choice` | Activa tool calling automático | False |
| `--tool-call-parser` | Parser de tool calls (ver [Tool Calling](tool-calling.md)) | None |

## Endpoints de la API

### Chat completions

```bash
POST /v1/chat/completions
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Sin streaming
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

# Con streaming
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Completions

```bash
POST /v1/completions
```

```python
response = client.completions.create(
    model="default",
    prompt="The capital of France is",
    max_tokens=50
)
```

### Models

```bash
GET /v1/models
```

Retorna los modelos disponibles.

### Embeddings

```bash
POST /v1/embeddings
```

```python
response = client.embeddings.create(
    model="mlx-community/multilingual-e5-small-mlx",
    input="Hello world"
)
print(response.data[0].embedding[:5])  # First 5 dimensions
```

Consulta la [Guía de Embeddings](embeddings.md) para más detalles.

### Health check

```bash
GET /health
```

Retorna el estado del servidor.

### Métricas

```bash
GET /metrics
```

Endpoint de scrape de Prometheus con métricas del servidor, caché, scheduler y solicitudes.
El endpoint está desactivado por defecto y se habilita con `--enable-metrics`.

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit \
  --enable-metrics
```

`/metrics` no requiere autenticación de forma intencional. Exponlo solo en una red de confianza o detrás de un proxy inverso o firewall que limite quién puede consultarlo.

### API de mensajes de Anthropic

```bash
POST /v1/messages
```

Endpoint compatible con Anthropic que permite que herramientas como Claude Code y OpenCode se conecten directamente a vllm-mlx. Internamente traduce las solicitudes de Anthropic al formato de OpenAI, ejecuta la inferencia a través del motor y convierte la respuesta de vuelta al formato de Anthropic.

Capacidades:
- Respuestas sin streaming y con streaming (SSE)
- Mensajes de sistema (cadena de texto simple o lista de bloques de contenido)
- Conversaciones multi-turno con mensajes de usuario y asistente
- Tool calling con bloques de contenido `tool_use` / `tool_result`
- Conteo de tokens para seguimiento de presupuesto
- Contenido multimodal (imágenes mediante bloques `source`)
- Detección de desconexión del cliente (retorna HTTP 499)
- Filtrado automático de tokens especiales en la salida en streaming

#### Sin streaming

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000", api_key="not-needed")

response = client.messages.create(
    model="default",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
# Response includes: response.id, response.model, response.stop_reason,
# response.usage.input_tokens, response.usage.output_tokens
```

#### Streaming

El streaming sigue el protocolo de eventos SSE de Anthropic. Los eventos se emiten en este orden:
`message_start` -> `content_block_start` -> `content_block_delta` (repetido) -> `content_block_stop` -> `message_delta` -> `message_stop`

```python
with client.messages.stream(
    model="default",
    max_tokens=256,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

#### Mensajes de sistema

Los mensajes de sistema pueden ser una cadena de texto simple o una lista de bloques de contenido:

```python
# Plain string
response = client.messages.create(
    model="default",
    max_tokens=256,
    system="You are a helpful coding assistant.",
    messages=[{"role": "user", "content": "Write a hello world in Python"}]
)

# List of content blocks
response = client.messages.create(
    model="default",
    max_tokens=256,
    system=[
        {"type": "text", "text": "You are a helpful assistant."},
        {"type": "text", "text": "Be concise in your answers."},
    ],
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
```

#### Tool calling

Define las herramientas con `name`, `description` e `input_schema`. El modelo retorna bloques de contenido `tool_use` cuando desea llamar a una herramienta. Envía los resultados de vuelta como bloques `tool_result`.

```python
# Step 1: Send request with tools
response = client.messages.create(
    model="default",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }]
)

# Step 2: Check if model wants to use tools
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}, Input: {block.input}, ID: {block.id}")
        # response.stop_reason will be "tool_use"

# Step 3: Send tool result back
response = client.messages.create(
    model="default",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": response.content},
        {"role": "user", "content": [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": "Sunny, 22C"
            }
        ]}
    ],
    tools=[{
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }]
)
print(response.content[0].text)  # "The weather in Paris is sunny, 22C."
```

Modos de selección de herramientas:

| `tool_choice` | Comportamiento |
|---------------|----------|
| `{"type": "auto"}` | El modelo decide si llamar herramientas (por defecto) |
| `{"type": "any"}` | El modelo debe llamar al menos una herramienta |
| `{"type": "tool", "name": "get_weather"}` | El modelo debe llamar la herramienta especificada |
| `{"type": "none"}` | El modelo no llamará ninguna herramienta |

#### Conversaciones multi-turno

```python
messages = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What's my name?"},
]

response = client.messages.create(
    model="default",
    max_tokens=100,
    messages=messages
)
```

#### Conteo de tokens

```bash
POST /v1/messages/count_tokens
```

Cuenta los tokens de entrada para una solicitud de Anthropic usando el tokenizador del modelo. Útil para el seguimiento de presupuesto antes de enviar una solicitud. Cuenta tokens de mensajes de sistema, mensajes de conversación, entradas de tool_use, contenido de tool_result y definiciones de herramientas (name, description, input_schema).

```python
import requests

resp = requests.post("http://localhost:8000/v1/messages/count_tokens", json={
    "model": "default",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "system": "You are helpful.",
    "tools": [{
        "name": "search",
        "description": "Search the web",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}
    }]
})
print(resp.json())  # {"input_tokens": 42}
```

#### Ejemplos con curl

Sin streaming:

```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Con streaming:

```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "max_tokens": 256,
    "stream": true,
    "messages": [{"role": "user", "content": "Tell me a joke"}]
  }'
```

Conteo de tokens:

```bash
curl http://localhost:8000/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
# {"input_tokens": 12}
```

#### Campos de la solicitud

| Campo | Tipo | Requerido | Valor por defecto | Descripción |
|-------|------|----------|---------|-------------|
| `model` | string | sí | - | Nombre del modelo (usa `"default"` para el modelo cargado) |
| `messages` | list | sí | - | Mensajes de conversación con `role` y `content` |
| `max_tokens` | int | sí | - | Número máximo de tokens a generar |
| `system` | string o list | no | null | Prompt de sistema (cadena o lista de bloques `{"type": "text", "text": "..."}`) |
| `stream` | bool | no | false | Activa el streaming SSE |
| `temperature` | float | no | 0.7 | Temperatura de muestreo (0.0 = determinista, 1.0 = creativo) |
| `top_p` | float | no | 0.9 | Umbral de nucleus sampling |
| `top_k` | int | no | null | Top-k sampling |
| `stop_sequences` | list | no | null | Secuencias que detienen la generación |
| `tools` | list | no | null | Definiciones de herramientas con `name`, `description`, `input_schema` |
| `tool_choice` | dict | no | null | Modo de selección de herramientas (`auto`, `any`, `tool`, `none`) |
| `metadata` | dict | no | null | Metadatos arbitrarios (se pasan sin ser usados por el servidor) |

#### Formato de respuesta

Respuesta sin streaming:

```json
{
  "id": "msg_abc123...",
  "type": "message",
  "role": "assistant",
  "model": "default",
  "content": [
    {"type": "text", "text": "Hello! How can I help?"}
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 12,
    "output_tokens": 8
  }
}
```

Cuando se llaman herramientas, `content` incluye bloques `tool_use` y `stop_reason` es `"tool_use"`:

```json
{
  "content": [
    {"type": "text", "text": "Let me check the weather."},
    {
      "type": "tool_use",
      "id": "call_abc123",
      "name": "get_weather",
      "input": {"city": "Paris"}
    }
  ],
  "stop_reason": "tool_use"
}
```

Razones de parada:

| `stop_reason` | Significado |
|---------------|---------|
| `end_turn` | El modelo terminó de forma natural |
| `tool_use` | El modelo quiere llamar una herramienta |
| `max_tokens` | Se alcanzó el límite de `max_tokens` |

#### Uso con Claude Code

Apunta Claude Code directamente a tu servidor vllm-mlx:

```bash
# Start the server
vllm-mlx serve mlx-community/Qwen3-Coder-Next-235B-A22B-4bit \
  --continuous-batching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# In another terminal, configure Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

### Estado del servidor

```bash
GET /v1/status
```

Endpoint de monitoreo en tiempo real que retorna estadísticas generales del servidor y detalles por solicitud. Útil para depurar el rendimiento, rastrear la eficiencia de la caché y monitorear la memoria GPU Metal.

```bash
curl -s http://localhost:8000/v1/status | python -m json.tool
```

Respuesta de ejemplo:

```json
{
  "status": "running",
  "model": "mlx-community/Qwen3-8B-4bit",
  "uptime_s": 342.5,
  "steps_executed": 1247,
  "num_running": 1,
  "num_waiting": 0,
  "total_requests_processed": 15,
  "total_prompt_tokens": 28450,
  "total_completion_tokens": 3200,
  "metal": {
    "active_memory_gb": 5.2,
    "peak_memory_gb": 8.1,
    "cache_memory_gb": 2.3
  },
  "cache": {
    "type": "memory_aware_cache",
    "entries": 5,
    "hit_rate": 0.87,
    "memory_mb": 2350
  },
  "requests": [
    {
      "request_id": "req_abc123",
      "phase": "generation",
      "tokens_per_second": 45.2,
      "ttft_s": 0.8,
      "progress": 0.35,
      "cache_hit_type": "prefix",
      "cached_tokens": 1200,
      "generated_tokens": 85,
      "max_tokens": 256
    }
  ]
}
```

Campos de la respuesta:

| Campo | Descripción |
|-------|-------------|
| `status` | Estado del servidor: `running`, `stopped` o `not_loaded` |
| `model` | Nombre del modelo cargado |
| `uptime_s` | Segundos desde que el servidor inició |
| `steps_executed` | Total de pasos de inferencia ejecutados |
| `num_running` | Número de solicitudes generando tokens actualmente |
| `num_waiting` | Número de solicitudes en cola para prefill |
| `total_requests_processed` | Total de solicitudes completadas desde el inicio |
| `total_prompt_tokens` | Total de tokens de prompt procesados desde el inicio |
| `total_completion_tokens` | Total de tokens de completion generados desde el inicio |
| `metal.active_memory_gb` | Memoria GPU Metal en uso actualmente (GB) |
| `metal.peak_memory_gb` | Uso pico de memoria GPU Metal (GB) |
| `metal.cache_memory_gb` | Uso de memoria de caché Metal (GB) |
| `cache` | Estadísticas de caché (tipo, entradas, tasa de aciertos, uso de memoria) |
| `requests` | Lista de solicitudes activas con detalles por solicitud |

Campos por solicitud en `requests`:

| Campo | Descripción |
|-------|-------------|
| `request_id` | Identificador único de la solicitud |
| `phase` | Fase actual: `queued`, `prefill` o `generation` |
| `tokens_per_second` | Rendimiento de generación para esta solicitud |
| `ttft_s` | Tiempo hasta el primer token (segundos) |
| `progress` | Porcentaje de completado (0.0 a 1.0) |
| `cache_hit_type` | Tipo de coincidencia en caché: `exact`, `prefix`, `supersequence`, `lcp` o `miss` |
| `cached_tokens` | Número de tokens servidos desde caché |
| `generated_tokens` | Tokens generados hasta ahora |
| `max_tokens` | Máximo de tokens solicitados |

## Tool Calling

Activa tool calling compatible con OpenAI con `--enable-auto-tool-choice`:

```bash
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

Usa la opción `--tool-call-parser` para seleccionar el parser adecuado para tu modelo:

| Parser | Modelos |
|--------|--------|
| `auto` | Detección automática (prueba todos los parsers) |
| `mistral` | Mistral, Devstral |
| `qwen` | Qwen, Qwen3 |
| `llama` | Llama 3.x, 4.x |
| `hermes` | Hermes, NousResearch |
| `deepseek` | DeepSeek V3, R1 |
| `kimi` | Kimi K2, Moonshot |
| `granite` | IBM Granite 3.x, 4.x |
| `nemotron` | NVIDIA Nemotron |
| `xlam` | Salesforce xLAM |
| `functionary` | MeetKai Functionary |
| `glm47` | GLM-4.7, GLM-4.7-Flash |

```python
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
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]
)

if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"{tc.function.name}: {tc.function.arguments}")
```

Consulta la [Guía de Tool Calling](tool-calling.md) para la documentación completa.

## Modelos de reasoning

Para modelos que muestran su proceso de pensamiento (Qwen3, DeepSeek-R1), usa `--reasoning-parser` para separar el reasoning de la respuesta final:

```bash
# Qwen3 models
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# DeepSeek-R1 models
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

La respuesta de la API incluye un campo `reasoning` con el proceso de pensamiento del modelo:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)

print(response.choices[0].message.reasoning)  # Step-by-step thinking
print(response.choices[0].message.content)    # Final answer
```

En streaming, los fragmentos de reasoning llegan primero, seguidos de los fragmentos de contenido:

```python
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.reasoning:
        print(f"[Thinking] {delta.reasoning}")
    if delta.content:
        print(delta.content, end="")
```

Consulta la [Guía de Modelos de Reasoning](reasoning.md) para todos los detalles.

## Salida estructurada (modo JSON)

Obliga al modelo a retornar JSON válido usando `response_format`:

### Modo JSON Object

Retorna cualquier JSON válido:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={"type": "json_object"}
)
# Output: {"colors": ["red", "blue", "green"]}
```

### Modo JSON Schema

Retorna JSON que coincide con un esquema específico:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "colors",
            "schema": {
                "type": "object",
                "properties": {
                    "colors": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["colors"]
            }
        }
    }
)
# Output validated against schema
data = json.loads(response.choices[0].message.content)
assert "colors" in data
```

### Ejemplo con curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "List 3 colors"}],
    "response_format": {"type": "json_object"}
  }'
```

## Ejemplos con curl

### Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Configuración de streaming

Controla el comportamiento del streaming con `--stream-interval`:

| Valor | Comportamiento |
|-------|----------|
| `1` (por defecto) | Envía cada token inmediatamente |
| `2-5` | Agrupa tokens antes de enviar |
| `10+` | Máximo rendimiento, salida en fragmentos más grandes |

```bash
# Smooth streaming
vllm-mlx serve model --continuous-batching --stream-interval 1

# Batched streaming (better for high-latency networks)
vllm-mlx serve model --continuous-batching --stream-interval 5
```

## Integración con Open WebUI

```bash
# 1. Start vllm-mlx server
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# 2. Start Open WebUI
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main

# 3. Open http://localhost:3000
```

## Despliegue en producción

### Con systemd

Crea `/etc/systemd/system/vllm-mlx.service`:

```ini
[Unit]
Description=vLLM-MLX Server
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching --use-paged-cache --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable vllm-mlx
sudo systemctl start vllm-mlx
```

### Configuración recomendada

Para producción con 50 o más usuarios concurrentes:

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120 \
  --port 8000
```
