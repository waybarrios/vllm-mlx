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

This is the eviction budget, not the full system RAM size. Leave headroom for:

- KV cache
- request batching
- OS / filesystem cache
- other colocated services

On a 128 GB machine, a practical starting point is often `80-100 GB`.

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
