# Warm Prompts

Pre-pobla el prefix cache al iniciar el servidor para que la **primera** solicitud
que envie un agent encuentre un cache caliente en lugar de pagar el prefill completo
de su system prompt de varios kilobytes.

## Cuándo usar esto

Las cargas de trabajo de agents, proxies hacia asistentes de código o razonamiento,
servidores MCP, orquestadores multi-agent, siempre envian el mismo system prompt.
Hoy, la primera solicitud desde un servidor frio paga el prefill completo de ese
sistema. En un modelo de miles de millones de parámetros eso equivale a varios
segundos de TTFT, justo cuando un usuario espera que su nuevo agent responda por
primera vez.

Si ya conoces los system prompts de tus agents al momento del despliegue, escríbelos
en un archivo JSON y apunta `--warm-prompts` hacia él. El servidor ejecuta un chat
completion de `max_tokens=1` para cada uno al inicio, el estado del KV cache queda
en el prefix cache, y la primera solicitud real coincide via strict-prefix.

Requiere `--continuous-batching` (el prefix cache vive ahí).

## Ejemplo rápido

```bash
# Write the agents you care about once
cat > ~/.config/vllm-mlx/agents.json <<'JSON'
[
  [{"role": "system", "content": "You are a code assistant..."}]
]
JSON

# Point the server at it
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --continuous-batching \
  --warm-prompts ~/.config/vllm-mlx/agents.json
```

Al iniciar verás:

```
[lifespan] Warm-up done (strict-prefix): 1 completed, 0 skipped,
           1431 prompt tokens in 0.2s
```

La primera solicitud real que comparte el system prompt calentado accede al cache
con `tokens_saved` cercano a la longitud del prompt de calentamiento.

## Formato del archivo

Una lista JSON de nivel superior. Cada entrada es a su vez una lista de mensajes
de chat, con la misma forma que `messages` en `/v1/chat/completions`.

```json
[
  [
    {"role": "system", "content": "You are a code assistant..."}
  ],
  [
    {"role": "system", "content": "You are a senior code reviewer..."}
  ],
  [
    {"role": "system", "content": "You are a planner..."},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "Hello, what are we planning?"}
  ]
]
```

Los system prompts de un solo mensaje son el caso más común. Los historiales
multi-turno son compatibles para escenarios en los que quieres calentar un inicio
de conversación específico (ejemplos few-shot, una persona de asistente fija).

## Dimensionamiento

Los warm prompts se procesan **de forma concurrente** via `asyncio.gather`, por lo
que N entradas lanzan N prefills concurrentes al inicio. Cada prefill asigna KV
cache según la longitud de su prompt.

**Recomendado: 1 a 3 entradas.** Eso cubre los caminos calientes de despliegues
típicos de agents (una persona por entrada). Un archivo warm-prompts muy grande en
un modelo con poca memoria puede agotar el espacio disponible en el arranque.

Si necesitas calentar decenas de personas, abre un issue con tu carga de trabajo y
podemos agregar un limite `--warm-prompts-concurrency=N`.

## Benchmarks

**Configuración.** M4 Max, 128 GB de memoria unificada. Dos servidores separados por
medición (frio vs caliente), arranque frio aislado. Conjunto de prompts `long`
(aprox. 2.5k tokens de usuario) antepuesto con un system prompt de aprox. 1.7k
tokens para coincidir con el warm prompt. `max_tokens=128`. bench-serve con
`--skip-preflight-token-count` para que el preflight de count_prompt_tokens no
contamine el cache.

| Model | conc | cold TTFT | warm TTFT | Speedup |
|-------|-----:|----------:|----------:|--------:|
| Qwen3-0.6B-8bit | 1 | 563 ms | 419 ms | 1.34x |
| Qwen3-0.6B-8bit | 4 | 1 723 ms | 1 282 ms | 1.34x |
| Qwen3-0.6B-8bit | 8 | 3 708 ms | 2 661 ms | 1.39x |
| Llama-3.2-3B-Instruct-4bit | 1 | 1 754 ms | 1 060 ms | 1.65x |
| Llama-3.2-3B-Instruct-4bit | 4 | 5 926 ms | 3 945 ms | 1.50x |
| Llama-3.2-3B-Instruct-4bit | 8 | 15 161 ms | 9 820 ms | 1.54x |
| Qwen3-4B-4bit | 1 | 4 937 ms | 2 191 ms | 2.25x |
| Qwen3-4B-4bit | 4 | 12 535 ms | 9 623 ms | 1.30x |
| Qwen3-4B-4bit | 8 | 38 148 ms | 23 878 ms | 1.60x |
| Qwen3.6-35B-A3B-4bit (MoE/hybrid) | 1 | 2 400 ms | 1 603 ms | 1.50x |
| Qwen3.6-35B-A3B-4bit | 4 | 8 735 ms | 6 054 ms | 1.44x |
| Qwen3.6-35B-A3B-4bit | 8 | 22 419 ms | 14 409 ms | 1.56x |

Las 12 configuraciones mejoran. Los ahorros de TTFT son mayores cuando la relación
prompt/total es más alta (conc=1, system prompt largo) y siguen siendo significativos
bajo carga concurrente.

**Generation tok/s** es neutral (dentro de +-5%) para los modelos densos.
Qwen3.6-35B-A3B (MoE) muestra una caida en decode del 20 al 35% con conc >= 4,
que parece ser una interacción del enrutamiento MoE con el scheduling en batch. Los
ahorros de TTFT siguen dominando la latencia extremo a extremo en cargas de trabajo
de agents, pero toma nota de esto si tu flujo es fuertemente decode-bound a alta
concurrencia.

## Cómo funciona

El calentamiento naive, renderizar la plantilla de chat con un mensaje de usuario de
relleno y cachear los tokens, no funciona para modelos híbridos SSM+attention
(Qwen3.5-MoE, Qwen3.6-MoE). Sus capas de cache incluyen estado SSM que no puede
recortarse, por lo que `memory_cache.py` deshabilita la coincidencia LCP. El
contenido de usuario de relleno diverge del contenido real del usuario y una entrada
cacheada a nivel de tokens ya no es un strict-prefix de ninguna solicitud real.

El calentador aquí renderiza la plantilla de chat **dos veces** con dos contenidos de
usuario distintos (`"__PROBE_A__"` y `"__PROBE_B__"`), encuentra la posición de
carácter donde las dos cadenas divergen y trunca el primer renderizado en ese limite.
Esa cadena truncada, todo lo que precede al punto donde se inserta el contenido del
usuario, es lo que se envía al motor.

Dado que el flujo de solicitudes reales del motor también renderiza la plantilla con
`tokenize=False` y luego deja que el tokenizador codifique el resultado, los tokens
del calentamiento tienen garantia de ser un strict-prefix de cualquier solicitud real
con un sistema coincidente e historial de chat vacío. Las coincidencias strict-prefix
funcionan en todo tipo de capas de cache, incluidos los flujos híbridos donde el LCP
está deshabilitado.

## Administración

### Limpiar el prefix cache en memoria

```bash
curl -X DELETE http://localhost:8000/v1/cache/prefix
```

Si el servidor se inició con `--warm-prompts`, el calentamiento se vuelve a ejecutar
en segundo plano después de la limpieza. La respuesta se devuelve de inmediato sin
esperar a que termine el re-calentamiento.

Respuesta:

```json
{"status": "cleared", "rewarm_scheduled": true}
```

### Inspeccionar el estado del cache

```bash
curl http://localhost:8000/v1/status | jq '.cache'
```

Tras el arranque con warm-prompts verás `entry_count > 0` antes de la primera
solicitud del usuario.

## Benchmark de tu propia configuración

Para medir el impacto en tu modelo y tus prompts, usa `bench-serve`:

```bash
# Cold: no warm-prompts
vllm-mlx serve MODEL --continuous-batching &
vllm-mlx bench-serve --prompts long --concurrency 1,4 \
  --system-prompt-file my-system.txt --tag cold \
  --output cold.csv --format csv

# Warm: same server config + --warm-prompts
vllm-mlx serve MODEL --continuous-batching \
  --warm-prompts ~/.config/vllm-mlx/agents.json &
vllm-mlx bench-serve --prompts long --concurrency 1,4 \
  --system-prompt-file my-system.txt --tag warm \
  --output warm.csv --format csv
```

`--skip-preflight-token-count` se habilita automáticamente cuando se usa
`--system-prompt-file`, por lo que el preflight de `count_prompt_tokens` no
contamina el cache. Compara `cold.csv` y `warm.csv` para tu carga de trabajo.
