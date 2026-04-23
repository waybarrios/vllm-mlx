# Tool Calling

vllm-mlx soporta tool calling compatible con OpenAI (function calling) con análisis automático para muchas familias de modelos populares.

## Inicio rápido

Activa el tool calling agregando la bandera `--enable-auto-tool-choice` al iniciar el servidor:

```bash
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

Luego usa herramientas con la API estándar de OpenAI:

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

## Parsers disponibles

Usa `--tool-call-parser` para seleccionar un tool parser según tu familia de modelos:

| Parser | Alias | Modelos | Formato |
|--------|-------|---------|---------|
| `auto` | | Cualquier modelo | Detecta el formato automáticamente (prueba todos los parsers) |
| `mistral` | | Mistral, Devstral | Arreglo JSON con `[TOOL_CALLS]` |
| `qwen` | `qwen3` | Qwen, Qwen3 | XML `<tool_call>` o `[Calling tool:]` |
| `llama` | `llama3`, `llama4` | Llama 3.x, 4.x | Etiquetas `<function=name>` |
| `hermes` | `nous` | Hermes, NousResearch | JSON `<tool_call>` dentro de XML |
| `deepseek` | `deepseek_v3`, `deepseek_r1` | DeepSeek V3, R1 | Delimitadores Unicode |
| `kimi` | `kimi_k2`, `moonshot` | Kimi K2, Moonshot | Tokens `<\|tool_call_begin\|>` |
| `granite` | `granite3` | IBM Granite 3.x, 4.x | `<\|tool_call\|>` o `<tool_call>` |
| `nemotron` | `nemotron3` | NVIDIA Nemotron | `<tool_call><function=...><parameter=...>` |
| `xlam` | | Salesforce xLAM | JSON con arreglo `tool_calls` |
| `functionary` | `meetkai` | MeetKai Functionary | Múltiples bloques de función |
| `glm47` | `glm4` | GLM-4.7, GLM-4.7-Flash | `<tool_call>` con XML `<arg_key>`/`<arg_value>` |

## Ejemplos por modelo

### Mistral / Devstral

```bash
# Devstral Small (optimizado para código y tool use)
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

## Parser automático

Si no sabes qué parser usar, el parser `auto` intenta detectar el formato de forma automática:

```bash
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --enable-auto-tool-choice --tool-call-parser auto
```

El parser automático prueba los formatos en este orden:
1. Mistral (`[TOOL_CALLS]`)
2. Qwen con corchetes (`[Calling tool:]`)
3. Nemotron (`<tool_call><function=...><parameter=...>`)
4. XML de Qwen/Hermes (`<tool_call>{...}</tool_call>`)
5. Llama (`<function=name>{...}</function>`)
6. JSON sin formato

## Streaming de tool calls

Los tool calls funcionan con streaming. La información del tool call se envía cuando el modelo termina de generarla:

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

## Manejo de resultados de herramientas

Después de recibir un tool call, ejecuta la función y devuelve el resultado:

```python
import json

# Primera solicitud: el modelo decide llamar a una herramienta
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[weather_tool]
)

# Obtener el tool call
tool_call = response.choices[0].message.tool_calls[0]
tool_call_id = tool_call.id
function_name = tool_call.function.name
arguments = json.loads(tool_call.function.arguments)

# Ejecutar la función (implementación propia)
result = get_weather(**arguments)  # {"temperature": 22, "condition": "sunny"}

# Enviar el resultado de vuelta al modelo
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

## Manejo de etiquetas de razonamiento

Los modelos que producen etiquetas de razonamiento `<think>...</think>` (como DeepSeek-R1, Qwen3, GLM-4.7) se manejan de forma automática. El parser elimina el contenido de reasoning antes de extraer los tool calls, por lo que las etiquetas de razonamiento nunca interfieren con el análisis de tool calls.

Esto funciona incluso cuando `<think>` fue inyectado en el prompt (etiquetas implícitas con solo un cierre `</think>`).

## Referencia de CLI

| Opción | Descripción |
|--------|-------------|
| `--enable-auto-tool-choice` | Activa el tool calling automático |
| `--tool-call-parser` | Selecciona el parser (ver tabla anterior) |

Consulta la [Referencia de CLI](../reference/cli.md) para todas las opciones.
