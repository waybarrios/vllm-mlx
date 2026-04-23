# Referencia de configuración

## Configuracion del servidor

### Opciones basicas

| Opcion | Descripcion | Valor por defecto |
|--------|-------------|---------|
| `--host` | Direccion del host del servidor | `127.0.0.1` |
| `--port` | Puerto del servidor | `8000` |
| `--max-tokens` | Maximo de tokens por defecto | `32768` |
| `--max-request-tokens` | Maximo de `max_tokens` aceptado de clientes de la API | `32768` |
| `--default-temperature` | Temperatura por defecto cuando no se especifica en la solicitud | None |
| `--default-top-p` | top_p por defecto cuando no se especifica en la solicitud | None |

### Opciones de seguridad

| Opcion | Descripcion | Valor por defecto |
|--------|-------------|---------|
| `--api-key` | Clave de API para autenticacion | None |
| `--rate-limit` | Solicitudes por minuto por cliente (0 = deshabilitado) | `0` |
| `--timeout` | Tiempo de espera de la solicitud en segundos | `300` |
| `--enable-metrics` | Expone métricas de Prometheus en `/metrics` | `false` |
| `--max-audio-upload-mb` | Tamano máximo de audio subido para `/v1/audio/transcriptions` | `25` |
| `--max-tts-input-chars` | Longitud máxima de texto aceptada por `/v1/audio/speech` | `4096` |

### Opciones de batching

| Opcion | Descripcion | Valor por defecto |
|--------|-------------|---------|
| `--continuous-batching` | Habilita el continuous batching | `false` |
| `--stream-interval` | Tokens por fragmento de streaming | `1` |
| `--max-num-seqs` | Maximo de secuencias concurrentes | `256` |

### Opciones de cache

| Opcion | Descripcion | Valor por defecto |
|--------|-------------|---------|
| `--cache-memory-mb` | Limite de memoria de cache en MB | Auto |
| `--cache-memory-percent` | Fraccion de RAM para cache | `0.20` |
| `--no-memory-aware-cache` | Usa cache de conteo de entradas heredado | `false` |
| `--use-paged-cache` | Habilita el KV cache paginado | `false` |
| `--paged-cache-block-size` | Tokens por bloque | `64` |
| `--max-cache-blocks` | Maximo de bloques | `1000` |

### Opciones de llamado a herramientas

| Opcion | Descripcion | Valor por defecto |
|--------|-------------|---------|
| `--enable-auto-tool-choice` | Habilita el llamado automático a herramientas | `false` |
| `--tool-call-parser` | Parser de llamados a herramientas (ver [Tool Calling](../guides/tool-calling.md)) | None |

### Opciones de razonamiento

| Opcion | Descripcion | Valor por defecto |
|--------|-------------|---------|
| `--reasoning-parser` | Parser para modelos de razonamiento (`qwen3`, `deepseek_r1`) | None |

### Opciones de embeddings

| Opcion | Descripcion | Valor por defecto |
|--------|-------------|---------|
| `--embedding-model` | Precarga un modelo de embeddings al iniciar | None |

### Opciones de MCP

| Opcion | Descripcion | Valor por defecto |
|--------|-------------|---------|
| `--mcp-config` | Ruta al archivo de configuración MCP | None |

## Configuracion de MCP

Crear `mcp.json`:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-name", "arg1"],
      "env": {
        "ENV_VAR": "value"
      }
    }
  }
}
```

### Opciones del servidor MCP

| Campo | Descripcion | Requerido |
|-------|-------------|----------|
| `command` | Comando ejecutable | Si |
| `args` | Argumentos del comando | Si |
| `env` | Variables de entorno | No |

## Opciones de solicitudes a la API

### Chat Completions

| Parametro | Descripcion | Valor por defecto |
|-----------|-------------|---------|
| `model` | Nombre del modelo | Requerido |
| `messages` | Mensajes del chat | Requerido |
| `max_tokens` | Maximo de tokens a generar | 256 |
| `temperature` | Temperatura de muestreo | Valor por defecto del modelo |
| `top_p` | Nucleus sampling | Valor por defecto del modelo |
| `stream` | Habilita el streaming | `true` |
| `stop` | Secuencias de detencion | None |
| `tools` | Definiciones de herramientas | None |
| `response_format` | Formato de salida (`json_object`, `json_schema`) | None |

### Opciones multimodales

| Parametro | Descripcion | Valor por defecto |
|-----------|-------------|---------|
| `video_fps` | Fotogramas por segundo | 2.0 |
| `video_max_frames` | Maximo de fotogramas | 32 |

## Variables de entorno

| Variable | Descripcion |
|----------|-------------|
| `VLLM_MLX_TEST_MODEL` | Modelo por defecto para pruebas |
| `HF_TOKEN` | Token de autenticacion de HuggingFace |
| `OPENAI_API_KEY` | Establecer a cualquier valor para compatibilidad con el SDK |

## Configuraciones de ejemplo

### Desarrollo (usuario único)

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit
```

### Produccion (multiples usuarios)

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --api-key your-secret-key \
  --rate-limit 60 \
  --port 8000
```

### Con llamado a herramientas

```bash
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral \
  --continuous-batching
```

### Con herramientas MCP

```bash
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --mcp-config mcp.json \
  --enable-auto-tool-choice \
  --tool-call-parser qwen \
  --continuous-batching
```

### Modelo de razonamiento

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit \
  --reasoning-parser qwen3 \
  --continuous-batching
```

### Con embeddings

```bash
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --embedding-model mlx-community/multilingual-e5-small-mlx \
  --continuous-batching
```

### Alto rendimiento

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --stream-interval 5 \
  --max-num-seqs 256
```
