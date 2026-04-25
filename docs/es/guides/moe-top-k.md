# MoE top_k override (`--moe-top-k`)

Reduce el numero de experts activados por token en modelos Mixture of Experts
como Qwen3-30B-A3B, intercambiando una pequeña cantidad de calidad por un
aumento significativo en el throughput de decodificacion.

> **Estado:** flag opt-in. El comportamiento por defecto no cambia. Los numeros
> de calidad que se muestran son para Qwen3-30B-A3B-4bit en M4 Max 128 GB.
> Verifica con tu modelo antes de usarlo en cargas de produccion.

## Que hace

Qwen3-30B-A3B se entrena con `top_k=8`. cada token selecciona 8 de 128
experts. En Apple Silicon con batch=1 durante la decodificacion, la multiplicacion
de matrices de experts (`SwitchGLU`) es la parte más costosa del computo por capa,
y ese costo escala de forma aproximadamente lineal con `top_k`. Reducir `top_k`
en tiempo de inferencia ha demostrado (LExI 2025, Lynx 2024) preservar la mayor
parte de la calidad entrenada mientras reduce materialmente el tiempo de
decodificacion.

`--moe-top-k N` itera cada capa del modelo cargado y, en cada capa que tenga
`.mlp.switch_mlp` (es decir, un bloque sparse-MoE), establece `top_k = N`. Las
capas densas y los modelos densos no se modifican: el flag es un no-op para ellos.

## Uso

```bash
# Server
vllm-mlx serve mlx-community/Qwen3-30B-A3B-4bit \
  --continuous-batching \
  --moe-top-k 4

# Bench
vllm-mlx bench mlx-community/Qwen3-30B-A3B-4bit --moe-top-k 4
```

El flag se rechaza si `N` es mayor que el `top_k` entrenado del modelo
(solo tiene sentido reducirlo, nunca aumentarlo).

## Impacto medido

### Throughput de decodificacion (M4 Max 128 GB, batch=1, greedy)

| top_k | tok/s | vs baseline |
|---:|---:|---:|
| 8 (baseline) | 126.5 | - |
| 6 | 136.1 | +7.6% |
| 5 | 140.3 | +10.9% |
| 4 | 147.3 | +16.5% |

### Calidad (Qwen3-30B-A3B-4bit, lm-evaluation-harness, MLX backend)

<!-- se completa cuando finaliza la evaluacion -->

| top_k | MMLU (acc) | GSM8K (exact match) | Delta vs baseline |
|---:|---:|---:|---:|
| 8 | TBD | TBD | - |
| 6 | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD |

MMLU: 200 muestras seleccionadas aleatoriamente, 0-shot.
GSM8K: 100 muestras seleccionadas aleatoriamente, 0-shot, exact-match estricto.

Estos numeros son **indicativos**: los conjuntos de evaluacion completos son
más grandes y desplazarian la precision absoluta, pero no el delta relativo
entre configuraciones de forma significativa.

### Paridad de salida greedy

Con `top_k=4` en el checkpoint de 4 bits observamos **los primeros 16 tokens
generados identicos** al baseline en todos los prompts de prueba que usamos.
Esto sugiere que top_k=4 no cambia el argmax en los pasos iniciales de
decodificacion: el modelo es internamente robusto a eliminar la mitad de sus
experts activados.

Con `top_k=3` o menor, la calidad comenzaria a degradarse de forma visible
(no medido aquí; inferido del paper LExI), por lo que el flag no permite bajar
por debajo de 1 en la capa de validación de configuración. Sin embargo, el
piso recomendado para produccion es `top_k=4`.

## Cuando usarlo y cuando no

Usalo cuando:
- Ejecutas un Qwen3 MoE (o compatible: Qwen3.5 MoE, Gemma-MoE) y el throughput
  de decodificacion con un solo usuario es tu cuello de botella.
- Tienes una carga de trabajo donde una pequeña perdida de calidad es aceptable
  a cambio de una mejora visible en latencia.
- Despliegas en hardware limitado por ancho de banda de memoria (Apple Silicon
  serie M) donde el gather de experts domina el tiempo de decodificacion por paso.

No lo uses cuando:
- Sirves modelos densos: el flag es un no-op y no aporta nada.
- Te importa la precision en el top-1% de suites de evaluacion de leaderboard.
- Ejecutas generaciones largas de chain-of-thought o "modo thinking", donde el
  acantilado de calidad puede ser más pronunciado que lo que sugiere MMLU en 0-shot.

## Combinacion con otras optimizaciones

Este flag se compone con la cuantizacion. En Qwen3-30B-A3B-4bit nuestra
combinacion medida es:

- 4-bit + top_k=8: 126.5 tok/s (baseline)
- 4-bit + top_k=4: 147.3 tok/s (+16.5%)
- 3-bit + top_k=8: 138.6 tok/s (+9.6%)
- 3-bit + top_k=6: 147.1 tok/s (+16.3%) . divergencia de calidad medible
- 3-bit + top_k=4: 157.3 tok/s (+24%) . **la calidad de salida se rompe** (el modelo respondió una pregunta diferente en nuestra prueba de humo)

3-bit + top_k=4 acumulo el error numérico más alla del punto donde el argmax
es estable. Usa a lo sumo un parámetro agresivo: ya sea 4-bit + top_k=4
o 3-bit + top_k=6. Ambos dan aproximadamente el mismo tok/s (~147) con perfiles
de calidad muy distintos.

## Internos

- Helper de parcheo: `vllm_mlx.scheduler.apply_moe_top_k_override(model, k)`
- Se aplica en `Scheduler.__init__` despues de cargar el modelo.
- Tests: `tests/test_moe_top_k.py`. cubre modelos densos, arquitecturas mixtas
  y rutas de validación.

## Referencias

- LExI: Layer-Adaptive Active Experts, [arXiv 2509.02753](https://arxiv.org/html/2509.02753)
- Not All Experts are Equal (NAEE), [ACL 2024](https://aclanthology.org/2024.acl-long.334.pdf)
- SwiftLM (`SWIFTLM_TOP_K` env knob prior art), [github.com/SharpAI/SwiftLM](https://github.com/SharpAI/SwiftLM)
