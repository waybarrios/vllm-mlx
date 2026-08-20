# Multi-Model Serving

`vllm-mlx` can serve a registry of named models behind one process and one OpenAI-compatible API surface.

This mode is designed for Apple Silicon machines where unified memory is the main constraint:

- models load lazily on first use
- idle models are evicted with an LRU policy under a memory budget
- contention can be configured to wait, fail fast, or preempt active models
- `/v1/models` reflects the configured registry instead of a single default model

## When to Use It

Use registry-backed serving when you want one server to expose multiple models such as:

- a small low-latency chat model
- a larger reasoning or coding model
- a multimodal model for image or video requests

Keep single-model serving when you want the smallest operational surface and the highest per-model simplicity.

## Start the Server

```bash
vllm-mlx serve --models-config /etc/vllm-mlx/models.yaml --host 0.0.0.0 --port 8000
```

Use `--memory-budget-gb` to override `manager.memory_budget_gb` (or the legacy
`manager.memory_budget`) for a particular launch. The CLI value takes precedence
over the YAML value and can supply the budget when the YAML field is absent.

You can still use global serve flags such as:

- `--api-key`
- `--rate-limit`
- `--timeout`
- `--default-temperature`
- `--default-top-p`
- `--reasoning-parser`
- `--enable-auto-tool-choice`
- `--tool-call-parser`

Do not combine `--models-config` with:

- a positional model argument
- `--served-model-name`

## Registry File

The registry is a YAML file with two top-level sections:

- `manager`: global budget and contention behavior
- `models`: named model entries that clients select via the OpenAI `model` field

Example:

```yaml
manager:
  memory_budget_gb: 100
  contention_policy:
    strategy: wait_then_preempt
    wait_timeout_s: 45
    preempt_after_s: 15

models:
  - name: fast
    path: /Users/david/ai-models/mlx_models/gemma-4-E2B-it-5bit
    preload: true
    continuous_batching: false
    estimated_memory_gb: 4

  - name: smart
    path: /Users/david/ai-models/mlx_models/Qwen3.5-27B-VLM-MTP-8bit
    continuous_batching: true
    enable_mtp: true
    estimated_memory_gb: 36

  - name: vision
    path: /Users/david/ai-models/mlx_models/gemma-4-31B-it-6bit
    mllm: true
    continuous_batching: true
    estimated_memory_gb: 44
```

## Manager Settings

### `memory_budget_gb`

Total resident-model budget for the registry manager.

**This budget counts model weights only.** It is the number the manager compares
against when deciding whether a new model fits or an idle one must be evicted.
It does not include, and does not reserve room for:

- KV cache
- activations during prefill and decode
- OS / filesystem cache
- other colocated services

On a 128 GB machine, a practical starting point is often `80-100 GB`.

For different host or launch profiles, pass `--memory-budget-gb` instead of
maintaining duplicate registry files. The override remains a weights-only
budget and does not change the Metal allocation ceiling described below.

### Budget vs. the Metal allocation ceiling

The manager budget and the MLX allocation ceiling are two separate numbers, and
the budget does not derive from the ceiling. The ceiling is installed at engine
start from `--gpu-memory-utilization`:

```
allocation_ceiling = gpu_memory_utilization x device_working_set_size
```

The weights *plus* the KV cache *plus* activations all have to fit under that
ceiling, while the budget only accounts for the weights. If the budget is set
above what is actually allocatable, the manager's arithmetic says N models fit,
it keeps them all resident, and MLX hits the ceiling — so you get a hard
out-of-memory failure instead of the graceful eviction the budget exists to
provide.

The invariant to maintain is:

```
memory_budget_gb  <=  gpu_memory_utilization x device_RAM
                      - KV/activation headroom
                      - prefix cache actually resident
```

The server reconciles the two process-wide terms at startup and logs them
together with the prefix-cache setting:

```
Registry memory budget: 68.0 GB of model weights; Metal allocation ceiling
64.0 GB (50% of 128.0 GB, from serve default); prefix-cache maximum
20.0 GB per continuous-batching engine (--cache-memory-mb, 2 of 3 entries)
```

When the weights budget alone does not fit below the ceiling, startup warns:

```
WARNING models-config manager.memory_budget_gb (68.0 GB) exceeds the Metal
allocation ceiling (64.0 GB). ...
```

This is a diagnostic, not a clamp — the server still starts with the budget you
configured. It is also a *necessary, not sufficient* condition: passing the
check does not mean you will not run out of memory, because the KV cache,
prefix cache and activations all come out of the same ceiling and are
workload-dependent. Treat the ceiling as an upper bound and leave real margin
below it.

Notes on how the check is computed:

- The Metal limit is installed only by continuous-batching entries — that is the
  one path calling `mx.set_memory_limit`, and simple-mode entries are not even
  constructed with a `gpu_memory_utilization`. The check therefore considers
  only the effective utilization of continuous-batching entries, taking the
  *lowest*, since each such load re-installs the process-wide limit. A
  `gpu_memory_utilization` set on a simple-mode entry has no effect on the
  ceiling and is ignored here.
- A registry with no continuous-batching entries gets **no** attributed ceiling:
  nothing installs one, so the report says so rather than deriving a figure from
  a value that is never applied. The serve default likewise only competes when
  some continuous-batching entry actually inherits it.
- The conflict check compares **only** the weights budget against the ceiling,
  because both are process-wide totals and therefore directly comparable.
- `--cache-memory-mb` is **not** subtracted from the ceiling. It is a per-engine
  maximum: it is cloned into each resident continuous-batching engine and
  allocated lazily, and simple-mode entries never receive it at all. Subtracting
  it once would understate capacity with one resident model and overstate it
  with several, so it is reported next to the ceiling rather than folded into
  it. It is reported only when it can actually bind — that is, for
  continuous-batching entries using the memory-aware prefix cache (not
  `--use-paged-cache`).
- A separate warning fires when `--cache-memory-mb` alone is at or above the
  ceiling, which is a configuration error in its own right.
- On hosts where MLX cannot report a Metal working-set size, the check reports
  that the budget could not be reconciled and issues no warning.

### `contention_policy`

Controls what happens when a request needs a model that does not currently fit.

Supported strategies:

- `fail`: return capacity failure immediately
- `wait`: wait for capacity to free up
- `preempt`: cancel active requests on other models and evict them
- `wait_then_fail`: wait up to `wait_timeout_s`, then fail
- `wait_then_preempt`: wait up to `preempt_after_s`, then start preempting, and stop waiting at `wait_timeout_s`

Recommended defaults:

- shared internal service: `wait_then_preempt`
- user-facing low-latency API: `wait_then_fail`
- strict isolation / no interruption: `wait`

## Model Entry Fields

Required:

- `name`: request-time model id
- one of `path`, `source`, or `model`

Optional:

- `preload`: load this model at startup
- `continuous_batching`: override the global mode for this model
- `mllm`: force multimodal loading when autodetect is not enough
- `enable_mtp`: enable native MTP for this model
- `prefill_step_size`
- `specprefill`
- `specprefill_threshold`
- `specprefill_keep_pct`
- `specprefill_draft_model`
- `stream_interval`
- `gpu_memory_utilization`
- `estimated_memory_gb`

## Sizing Rules

For deterministic eviction behavior:

- local models should have real weight files on disk
- non-local model ids should set `estimated_memory_gb`

If a registry entry points at a non-local source and no `estimated_memory_gb` is provided, startup will reject the config. This prevents the manager from making bad eviction decisions from guesswork.

Both sizing paths are **weight estimates, not total runtime memory**:

- for a local source, the estimate is the summed on-disk size of the entry's
  `.safetensors` / `.gguf` files
- for a declared model id, the estimate is the operator-supplied
  `estimated_memory_gb`

Neither includes KV cache or activations, so a model's real peak footprint is
larger than the number the manager charges against `memory_budget_gb`. Size the
budget with that gap in mind — see
[Budget vs. the Metal allocation ceiling](#budget-vs-the-metal-allocation-ceiling).

## Request Routing

Clients select a registry entry through the normal OpenAI `model` field:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="smart",
    messages=[{"role": "user", "content": "Explain speculative decoding."}],
)
```

If the requested model is not registered, the server returns `404` and lists the configured model ids.

## Operational Checks

### Inspect registry state

```bash
curl http://localhost:8000/v1/models
```

Registry-backed responses include the configured model ids and current state such as:

- `loaded`
- `loading`
- `unloaded`
- `preempting`

### Verify a cold-load path

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "fast",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 32
  }'
```

Then repeat with a second model id to verify:

- lazy load works
- the memory budget is enforced
- the selected contention policy behaves as expected

## Recommended Rollout

1. Start with local-disk model paths, not remote model ids.
2. Set `estimated_memory_gb` for every large model, even when local, so your operational budget stays explicit.
3. Preload only the model that must be instantly available.
4. Verify `/v1/models` before exposing the endpoint to shared traffic.
5. Exercise the configured contention strategy under load before production cutover.

## Failure Modes to Expect

- Bad or missing `estimated_memory_gb` on non-local sources: config load failure
- Too-small `memory_budget_gb`: repeated capacity failures or unnecessary preemption
- Too-large `memory_budget_gb` relative to `--gpu-memory-utilization`: MLX
  out-of-memory instead of eviction (the startup log warns about this)
- Over-aggressive `preempt` policy: active requests get cancelled during model swaps
- Too many `preload: true` entries: startup load storm and immediate budget pressure

## Choosing Per-Model Overrides

Use global defaults for the common case, then override only the model-specific performance knobs that materially differ.

Good candidates for per-model overrides:

- `continuous_batching`
- `enable_mtp`
- `mllm`
- `prefill_step_size`
- `stream_interval`

Keep these global unless you have a strong reason not to:

- auth
- rate limits
- request timeout
- reasoning parser selection
- tool parser selection
- manager memory budget / contention policy
