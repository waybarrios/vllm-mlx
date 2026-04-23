# Referencia de CLI

## Resumen de comandos

| Comando | Descripcion |
|---------|-------------|
| `vllm-mlx serve` | Inicia el servidor compatible con OpenAI |
| `vllm-mlx-bench` | Ejecuta benchmarks de rendimiento |
| `vllm-mlx-chat` | Inicia la interfaz de chat con Gradio |

## `vllm-mlx serve`

Inicia el servidor de API compatible con OpenAI.

### Uso

```bash
vllm-mlx serve <model> [options]
```

### Opciones

| Opcion | Descripcion | Por defecto |
|--------|-------------|-------------|
| `--served-model-name` | Nombre personalizado del modelo expuesto a traves de la API de OpenAI. Si no se especifica, se usa la ruta del modelo como nombre. | None |
| `--port` | Puerto del servidor | 8000 |
| `--host` | Host del servidor | 127.0.0.1 |
| `--api-key` | Clave de API para autenticacion | None |
| `--rate-limit` | Solicitudes por minuto por cliente (0 = desactivado) | 0 |
| `--timeout` | Tiempo limite de solicitud en segundos | 300 |
| `--enable-metrics` | Expone métricas de Prometheus en `/metrics` | False |
| `--continuous-batching` | Activa continuous batching para multiples usuarios | False |
| `--cache-memory-mb` | Limite de memoria para cache en MB | Auto |
| `--cache-memory-percent` | Fraccion de RAM para cache | 0.20 |
| `--no-memory-aware-cache` | Usa cache legacy basado en conteo de entradas | False |
| `--use-paged-cache` | Activa el KV cache paginado | False |
| `--max-tokens` | Maximo de tokens por defecto | 32768 |
| `--max-request-tokens` | Maximo de `max_tokens` aceptado desde clientes de la API | 32768 |
| `--stream-interval` | Tokens por fragmento de streaming | 1 |
| `--mcp-config` | Ruta al archivo de configuración de MCP | None |
| `--paged-cache-block-size` | Tokens por bloque de cache | 64 |
| `--max-cache-blocks` | Maximos bloques de cache | 1000 |
| `--max-num-seqs` | Maximo de secuencias concurrentes | 256 |
| `--default-temperature` | Temperatura por defecto cuando no se especifica en la solicitud | None |
| `--default-top-p` | top_p por defecto cuando no se especifica en la solicitud | None |
| `--max-audio-upload-mb` | Tamano máximo de audio subido para `/v1/audio/transcriptions` | 25 |
| `--max-tts-input-chars` | Longitud máxima de texto aceptada por `/v1/audio/speech` | 4096 |
| `--reasoning-parser` | Parser para modelos de reasoning (`qwen3`, `deepseek_r1`) | None |
| `--embedding-model` | Pre-carga un modelo de embeddings al iniciar | None |
| `--enable-auto-tool-choice` | Activa tool calling automático | False |
| `--tool-call-parser` | Parser de tool calling (`auto`, `mistral`, `qwen`, `llama`, `hermes`, `deepseek`, `kimi`, `granite`, `nemotron`, `xlam`, `functionary`, `glm47`) | None |

### Ejemplos

```bash
# Modo simple (usuario único, máximo rendimiento)
# La ruta del modelo se usa como nombre en la API de OpenAI (ej. model="mlx-community/Llama-3.2-3B-Instruct-4bit")
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit

Model will show up as 'mlx-community/Llama-3.2-3B-Instruct-4bit' in the `/v1/models` API endpoint. View with `curl http://localhost:8000/v1/models` or similar.

# Con un nombre de modelo personalizado en la API (el modelo se accede como "my-model" via la API de OpenAI)
# --served-model-name establece el nombre que los clientes deben usar al llamar a la API (ej. model="my-model")
vllm-mlx serve --served-model-name my-model mlx-community/Llama-3.2-3B-Instruct-4bit
# Note: Model will show up as 'my-model' in the `/v1/models` API endpoint.

# Continuous batching (multiples usuarios)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

# Con limite de memoria para modelos grandes
vllm-mlx serve mlx-community/GLM-4.7-Flash-4bit \
  --continuous-batching \
  --cache-memory-mb 2048

# Produccion con paged cache
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --port 8000

# Con herramientas MCP
vllm-mlx serve mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

# Modelo multimodal
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit

# Modelo de reasoning (separa el pensamiento de la respuesta)
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# Modelo de reasoning DeepSeek
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1

# Tool calling con Mistral/Devstral
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral

# Tool calling con Granite
vllm-mlx serve mlx-community/granite-4.0-tiny-preview-4bit \
  --enable-auto-tool-choice --tool-call-parser granite

# Con autenticacion por clave de API
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --api-key your-secret-key

# Exponer métricas de Prometheus
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --enable-metrics

# Configuracion de produccion con opciones de seguridad
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120 \
  --continuous-batching
```

### Seguridad

Cuando se establece `--api-key`, todas las solicitudes a la API requieren el encabezado `Authorization: Bearer <api-key>`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key"  # Must match --api-key
)
```

O con curl:

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-secret-key"
```

## `vllm-mlx-bench`

Ejecuta benchmarks de rendimiento.

### Uso

```bash
vllm-mlx-bench --model <model> [options]
```

### Opciones

| Opcion | Descripcion | Por defecto |
|--------|-------------|-------------|
| `--model` | Nombre del modelo | Requerido |
| `--prompts` | Numero de prompts | 5 |
| `--max-tokens` | Maximo de tokens por prompt | 256 |
| `--quick` | Modo de benchmark rápido | False |
| `--video` | Ejecutar benchmark de video | False |
| `--video-url` | URL de video personalizada | None |
| `--video-path` | Ruta de video personalizada | None |

### Ejemplos

```bash
# Benchmark de LLM
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit

# Benchmark rápido
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --quick

# Benchmark de imagenes (deteccion automática para modelos VLM)
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Benchmark de video
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --video

# Video personalizado
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit \
  --video --video-url https://example.com/video.mp4
```

## `vllm-mlx-chat`

Inicia la interfaz de chat con Gradio.

### Uso

```bash
vllm-mlx-chat --served-model-name <model-name> [options]
```

### Opciones

| Opcion | Descripcion | Por defecto |
|--------|-------------|-------------|
| `--model` | Nombre del modelo | Requerido |
| `--port` | Puerto de Gradio | 7860 |
| `--text-only` | Desactiva el modo multimodal | False |

### Ejemplos

```bash
# Chat multimodal (texto + imagenes + video)
vllm-mlx-chat --served-model-name mlx-community/Qwen3-VL-4B-Instruct-3bit

# Chat solo de texto
vllm-mlx-chat --served-model-name mlx-community/Llama-3.2-3B-Instruct-4bit --text-only
```

## Variables de entorno

| Variable | Descripcion |
|----------|-------------|
| `VLLM_MLX_TEST_MODEL` | Modelo para pruebas |
| `HF_TOKEN` | Token de HuggingFace |
