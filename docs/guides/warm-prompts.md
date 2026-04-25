# Warm Prompts

Pre-populate the prefix cache at server startup so the **first** request
an agent sends hits a warm cache instead of paying the full prefill for
its multi-kilobyte system prompt.

## When to use this

Agent workloads — proxies to coding/reasoning assistants, MCP servers,
multi-agent orchestrators — always send the same system prompt. Today
the first request from a cold server pays the full prefill for that
system. On a multi-billion-parameter model that is several seconds of
TTFT, landing exactly when a user is waiting for their new agent to
respond for the first time.

If you already know your agents' system prompts at deploy time, write
them to a JSON file and point `--warm-prompts` at it. The server
runs a `max_tokens=1` chat completion for each at startup, the KV
state lands in the prefix cache, and the first real request matches
via strict-prefix.

Requires `--continuous-batching` (the prefix cache lives there).

## Quick example

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

On start you'll see:

```
[lifespan] Warm-up done (strict-prefix): 1 completed, 0 skipped,
           1431 prompt tokens in 0.2s
```

The first real request that shares the warmed system prompt hits the
cache with `tokens_saved` close to the warm-up prompt length.

## File format

A top-level JSON list. Each entry is itself a list of chat messages —
same shape as `messages` in `/v1/chat/completions`.

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

Single-message system prompts are the common case. Multi-turn histories
are supported for scenarios where you want to warm a specific
conversation start (few-shot examples, a running assistant persona).

## Sizing

Warm-up prompts are processed **concurrently** via `asyncio.gather`,
so N entries fire N concurrent prefills at startup. Each prefill
allocates KV cache for its prompt length.

**Recommended: 1–3 entries.** That covers the hot paths for typical
agent deployments (one persona per entry). A very large warm-prompts
file on a memory-tight model can exhaust headroom at boot.

If you need to warm dozens of personas, open an issue with your
workload and we can add a `--warm-prompts-concurrency=N` cap.

## Benchmarks

**Setup.** M4 Max, 128 GB unified memory. Two separate servers per
measurement (cold vs warm), isolated cold start. `long` prompt set
(~2.5k user tokens) prepended with a ~1.7k-token system prompt to
match the warm-up prompt. `max_tokens=128`. bench-serve with
`--skip-preflight-token-count` so the count_prompt_tokens preflight
does not pollute the cache.

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

All 12 configurations improve. TTFT savings are largest when the
prompt-to-total ratio is highest (conc=1, long system prompt) and
still meaningful under concurrent load.

**Generation tok/s** is neutral (within ±5%) for the dense models.
Qwen3.6-35B-A3B (MoE) shows a 20–35% decode drop at conc ≥ 4 that
appears to be MoE routing interaction with batched scheduling. TTFT
savings still dominate end-to-end latency on agent workloads, but
note this if your workflow is heavily decode-bound at high
concurrency.

## How it works

The naive warm-up — render the chat template with a placeholder user
message and cache the tokens — does not work for hybrid SSM+attention
models (Qwen3.5-MoE, Qwen3.6-MoE). Their cache layers include SSM
state that cannot be trimmed, so `memory_cache.py` disables LCP
matching. The placeholder user content diverges from real user
content, and a tokens-level cached entry is no longer a strict prefix
of any real request.

The warmer here renders the chat template **twice** with two distinct
user contents (`"__PROBE_A__"` and `"__PROBE_B__"`), finds the
character position where the two strings diverge, and truncates the
first rendering at that boundary. That truncated string — everything
up to the point where user content gets inserted — is what goes to
the engine.

Because the engine's real-request path also renders the template with
`tokenize=False` and then lets the tokenizer encode the result, the
warm-up's tokens are guaranteed to be a strict prefix of any real
request with a matching system and empty chat history. Strict prefix
matches work on every cache layer type, including the hybrid paths
where LCP is disabled.

## Admin

### Clear the in-memory prefix cache

```bash
curl -X DELETE http://localhost:8000/v1/cache/prefix
```

If the server was started with `--warm-prompts`, the warm-up re-runs
in the background after clear. The response returns immediately without
waiting for re-warm.

Response:

```json
{"status": "cleared", "rewarm_scheduled": true}
```

### Inspect cache state

```bash
curl http://localhost:8000/v1/status | jq '.cache'
```

After startup with warm-prompts you will see `entry_count > 0` before
the first user request.

## Benchmarking your own setup

To measure the impact on your model and prompts, use `bench-serve`:

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

`--skip-preflight-token-count` is auto-enabled when
`--system-prompt-file` is set, so the `count_prompt_tokens` preflight
does not pollute the cache. Compare `cold.csv` and `warm.csv` for
your workload.
