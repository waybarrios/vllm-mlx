# Modelos de reasoning

vllm-mlx admite modelos de reasoning que muestran su proceso de thinking antes de dar una respuesta. Modelos como Qwen3 y DeepSeek-R1 envuelven su reasoning en etiquetas `<think>...</think>`, y vllm-mlx puede analizar estas etiquetas para separar el reasoning de la respuesta final.

## Por que usar el reasoning parser?

Cuando un modelo de reasoning genera salida, normalmente luce asi:

```
<think>
Let me analyze this step by step.
First, I need to consider the constraints.
The answer should be a prime number less than 10.
Checking: 2, 3, 5, 7 are all prime and less than 10.
</think>
The prime numbers less than 10 are: 2, 3, 5, 7.
```

Sin el reasoning parser, obtienes la salida cruda con las etiquetas incluidas. Con el reasoning parser habilitado, el proceso de thinking y la respuesta final se separan en campos distintos dentro de la respuesta de la API.

## Primeros pasos

### Iniciar el servidor con el reasoning parser

```bash
# For Qwen3 models
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# For DeepSeek-R1 models
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

### Formato de respuesta de la API

Cuando el reasoning parser esta habilitado, la respuesta de la API incluye un campo `reasoning`:

**Respuesta sin streaming:**

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

**Respuesta con streaming:**

Los fragmentos se envian por separado para el reasoning y el contenido. Durante la fase de reasoning, los fragmentos tienen `reasoning` con valor. Cuando el modelo pasa a la respuesta final, los fragmentos tienen `content` con valor:

```json
{"delta": {"reasoning": "Let me analyze"}}
{"delta": {"reasoning": " this step by step."}}
{"delta": {"reasoning": "\nFirst, I need to"}}
...
{"delta": {"content": "The prime"}}
{"delta": {"content": " numbers less than 10"}}
{"delta": {"content": " are: 2, 3, 5, 7."}}
```

## Uso con el SDK de OpenAI

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

### Streaming con reasoning

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

## Parsers disponibles

### Parser de Qwen3 (`qwen3`)

Para modelos Qwen3 que usan etiquetas explicitas `<think>` y `</think>`.

- Requiere **ambas** etiquetas, la de apertura y la de cierre
- Si faltan las etiquetas, la salida se trata como contenido regular
- Recomendado para: Qwen3-0.6B, Qwen3-4B, Qwen3-8B y modelos similares

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

### Parser de DeepSeek-R1 (`deepseek_r1`)

Para modelos DeepSeek-R1 que pueden omitir la etiqueta de apertura `<think>`.

- Mas permisivo que el parser de Qwen3
- Maneja casos donde `<think>` es implicita
- El contenido antes de `</think>` se trata como reasoning incluso sin `<think>`

```bash
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

## Como funciona

El reasoning parser usa deteccion basada en texto para identificar etiquetas de thinking en la salida del modelo. Durante el streaming, rastrea la posicion actual en la salida para enrutar correctamente cada token a `reasoning` o a `content`.

```
Model Output:        <think>Step 1: analyze...</think>The answer is 42.
                     ├─────────────────────┤├─────────────────────┤
Parsed:              │     reasoning       ││       content       │
                     └─────────────────────┘└─────────────────────┘
```

El parsing no tiene estado y usa el texto acumulado para determinar el contexto, lo que lo hace robusto para escenarios de streaming donde los tokens pueden llegar en fragmentos arbitrarios.

## Consejos para mejores resultados

### Prompting

Los modelos de reasoning funcionan mejor cuando se les anima a pensar paso a paso:

```python
messages = [
    {"role": "system", "content": "Think through problems step by step before answering."},
    {"role": "user", "content": "What is 17 × 23?"}
]
```

### Manejo del reasoning ausente

Algunos prompts pueden no activar el reasoning. En esos casos, `reasoning` sera `None` y toda la salida va a `content`:

```python
message = response.choices[0].message
if message.reasoning:
    print(f"Model's thought process: {message.reasoning}")
print(f"Answer: {message.content}")
```

### Temperatura y reasoning

Las temperaturas más bajas tienden a producir patrones de reasoning más consistentes:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    temperature=0.3  # More focused reasoning
)
```

## Compatibilidad con versiones anteriores

Cuando no se especifica `--reasoning-parser`, el servidor se comporta como antes:
- Las etiquetas de thinking se incluyen en el campo `content`
- No se agrega el campo `reasoning` a las respuestas

Esto garantiza que las aplicaciones existentes sigan funcionando sin cambios.

## Ejemplo: solucionador de problemas matematicos

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

## Ejemplos con curl

### Sin streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 15% of 80?"}]
  }'
```

### Con streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 15% of 80?"}],
    "stream": true
  }'
```

## Solucion de problemas

### No aparece el campo reasoning en la respuesta

- Asegurate de haber iniciado el servidor con `--reasoning-parser`
- Verifica que el modelo realmente use etiquetas de thinking (no todos los prompts activan el reasoning)

### El reasoning aparece en content

- Es posible que el modelo no este usando el formato de etiquetas esperado
- Prueba un parser diferente (`qwen3` vs `deepseek_r1`)

### Reasoning truncado

- Aumenta `--max-tokens` si el modelo esta alcanzando el limite de tokens a mitad del thinking

## Relacionado

- [Modelos admitidos](../reference/models.md) - Modelos que admiten reasoning
- [Configuracion del servidor](server.md) - Todas las opciones del servidor
- [Referencia de CLI](../reference/cli.md) - Opciones de línea de comandos
