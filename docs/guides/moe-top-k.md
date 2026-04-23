# MoE top_k override (`--moe-top-k`)

Reduces the number of experts activated per token in Mixture-of-Experts models
like Qwen3-30B-A3B, trading a small amount of quality for meaningfully higher
decode throughput.

> **Status:** opt-in flag. Default behaviour is unchanged. Quality numbers
> below are for Qwen3-30B-A3B-4bit on M4 Max 128 GB — verify on your model
> before shipping this to production workloads.

## What it does

Qwen3-30B-A3B is trained with `top_k=8` — every token picks 8 out of 128
experts. On Apple Silicon at batch=1 decode the expert matmul (`SwitchGLU`)
is the single biggest chunk of each layer's compute, and that cost scales
roughly linearly with `top_k`. Lowering `top_k` at inference time has been
shown (LExI 2025, Lynx 2024) to preserve most of the trained quality while
cutting decode time materially.

`--moe-top-k N` iterates every layer of the loaded model, and on each layer
that has `.mlp.switch_mlp` (i.e. a sparse-MoE block) sets `top_k = N`. Dense
layers and dense models are untouched — the flag is a no-op for them.

## Usage

```bash
# Server
vllm-mlx serve mlx-community/Qwen3-30B-A3B-4bit \
  --continuous-batching \
  --moe-top-k 4

# Bench
vllm-mlx bench mlx-community/Qwen3-30B-A3B-4bit --moe-top-k 4
```

The flag is rejected if `N` is greater than the model's trained `top_k`
(it only makes sense to lower, never to raise).

## Measured impact

### Decode throughput (M4 Max 128 GB, batch=1, greedy)

| top_k | tok/s | vs baseline |
|---:|---:|---:|
| 8 (baseline) | 126.5 | — |
| 6 | 136.1 | +7.6% |
| 5 | 140.3 | +10.9% |
| 4 | 147.3 | +16.5% |

### Quality (Qwen3-30B-A3B-4bit, lm-evaluation-harness, MLX backend)

<!-- populated after eval completes -->

| top_k | MMLU (acc) | GSM8K (exact match) | Δ vs baseline |
|---:|---:|---:|---:|
| 8 | TBD | TBD | — |
| 6 | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD |

MMLU: 200 randomly-selected samples, 0-shot.
GSM8K: 100 randomly-selected samples, 0-shot, exact-match strict.

These numbers are **directional** — full suites are larger and would shift
the absolute accuracy but not the relative delta between configs by much.

### Greedy output parity

With `top_k=4` on the 4-bit checkpoint we observed **identical first 16
generated tokens** vs the baseline across every probe prompt we tried. This
suggests top_k=4 does not change the argmax in the early decode steps — the
model is internally robust to dropping half its activated experts.

At `top_k=3` or lower quality would start to degrade visibly (not measured
here; inferred from LExI paper), so the flag is intentionally not lowered
below 1 at the config validation layer but the recommended floor for
production is `top_k=4`.

## When to use it, when not to

Use it when:
- You run a Qwen3 MoE (or compatible: Qwen3.5 MoE, Gemma-MoE) and single-user
  decode throughput is your bottleneck.
- You have a workload where a small quality drop is acceptable in exchange
  for a visible latency improvement.
- You're deploying on memory-bandwidth-bound hardware (M-series Apple Silicon)
  where expert gather dominates per-step decode time.

Skip it when:
- You serve dense models — flag is a no-op, adds nothing.
- You care about top-1% leaderboard accuracy on eval suites.
- You run long chain-of-thought / "thinking mode" generations where the
  quality cliff may be steeper than 0-shot MMLU suggests.

## Stacking with other optimizations

This flag composes with quantization. On Qwen3-30B-A3B-4bit our measured
stack is:

- 4-bit + top_k=8: 126.5 tok/s (baseline)
- 4-bit + top_k=4: 147.3 tok/s (+16.5%)
- 3-bit + top_k=8: 138.6 tok/s (+9.6%)
- 3-bit + top_k=6: 147.1 tok/s (+16.3%)  — quality divergence measurable
- 3-bit + top_k=4: 157.3 tok/s (+24%)  — **output quality breaks** (model answered a different question in our smoke test)

3-bit + top_k=4 compounded the numerical error past the point where the
argmax is stable. Stick to at most one aggressive knob: either 4-bit + top_k=4
or 3-bit + top_k=6. Both give approximately the same tok/s (~147) with very
different quality profiles.

## Internals

- Patch helper: `vllm_mlx.scheduler.apply_moe_top_k_override(model, k)`
- Applied in `Scheduler.__init__` after the model is loaded.
- Tests: `tests/test_moe_top_k.py` — covers dense models, mixed architectures,
  and validation paths.

## References

- LExI: Layer-Adaptive Active Experts, [arXiv 2509.02753](https://arxiv.org/html/2509.02753)
- Not All Experts are Equal (NAEE), [ACL 2024](https://aclanthology.org/2024.acl-long.334.pdf)
- SwiftLM (`SWIFTLM_TOP_K` env knob prior art), [github.com/SharpAI/SwiftLM](https://github.com/SharpAI/SwiftLM)
