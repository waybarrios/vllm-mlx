# Model And Serve Profile Matrix

Serve profiles keep runtime flags stable while you vary the model and request
payload. Use them from a source checkout when you need a repeatable local
backend for manual validation, client setup, or latency measurements.

The model id remains an operator choice. Pick an MLX model that fits local
memory and matches the workload, then choose the narrowest profile that exposes
the behavior you need.

## Serve Profiles

| Profile | Best use | Main serve flags |
|---|---|---|
| `text-default` | Daily local text serving | `--runtime-mode simple --cache-strategy auto` |
| `text-deterministic` | Reproduction runs and bug triage | `--deterministic` |
| `text-tools` | OpenAI-compatible tool calling | `--runtime-mode auto --cache-strategy auto --enable-auto-tool-choice --tool-call-parser auto` |
| `text-json` | JSON extraction and longer structured-output requests | `--runtime-mode simple --cache-strategy auto --timeout 900` |
| `mllm-default` | Single-user image or mixed media chat | `--runtime-mode simple --cache-strategy auto --mllm` |
| `mllm-correctness` | Multimodal concurrency with correctness bias | `--runtime-mode auto --cache-strategy auto --mllm --batch-divergence-monitor --batch-divergence-action serialize` |

## Common Starts

```bash
scripts/serve_profile.sh text-default mlx-community/Qwen3-4B-Instruct-2507-4bit
scripts/serve_profile.sh text-tools mlx-community/Qwen3-4B-Instruct-2507-4bit
scripts/serve_profile.sh text-json mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
scripts/serve_profile.sh mllm-default mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit
```

`text-deterministic` is the right first stop for reports where output drift
would obscure the failure. `text-tools` is the right profile when the request
includes `tools` or `tool_choice`. `mllm-correctness` trades throughput for a
safer multimodal concurrency posture.

## Measurement Harnesses

The checkout includes small harnesses for repeatable local checks against a
running server:

| Script | Measures |
|---|---|
| `scripts/serve_profile_benchmark.py` | Chat-completions throughput and usage totals |
| `scripts/streaming_latency_harness.py` | TTFT, total stream time, token counts, and inter-token latency |
| `scripts/prefix_reuse_harness.py` | Cold vs warm repeated-prefix request timing |
| `scripts/batch_invariance_harness.py` | Serial vs concurrent deterministic-output agreement |

Each harness accepts `--base-url`, `--model`, and an optional JSON output path.
Run `--help` on the script for the full argument list.

## Request Defaults

Profile defaults are intentionally conservative. Set `temperature`, `top_p`,
`max_tokens`, `enable_thinking`, and `response_format` in the request payload
when a test depends on those values. Keep the serve profile fixed while
comparing request-level behavior.

## Related Guides

- [Client Compatibility](client-compatibility.md)
- [OpenAI-Compatible Server](server.md)
- [Tool Calling](tool-calling.md)
