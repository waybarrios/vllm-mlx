# Block Speculative Decoding (DSpark)

`--spec-draft dspark` adds block-draft speculative decoding for NemotronH targets
(Nemotron 3.5 Lightning 30B-A3B) on the continuous-batching text path. It drafts up to
7 tokens per round with NVIDIA's [DSpark](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark)
drafter and verifies the whole block in one target forward.

This page is the design note and the measurements behind
[waybarrios/vllm-mlx#699](https://github.com/waybarrios/vllm-mlx/issues/699). Read the
**Status** section first: the implementation is correct and reaches NVIDIA's acceptance
figures, but on today's MLX kernels it decodes at 0.7–0.8× plain greedy speed. The
bottleneck is measured and named below; it is a kernel-level property of MoE verify
width on MLX, not the drafter.

## Quick start

```bash
# 1. Drafter weights into the target's model directory (one of two artifacts, see below)
python scripts/prepare_dspark_weights.py --mlx-model-path <nemotron-6bit-dir>

# 2. Serve. Greedy, single-stream requests engage the drafter; everything else
#    runs plain decode on the same server.
vllm-mlx serve <nemotron-6bit-dir> --continuous-batching \
    --spec-draft dspark --spec-num-draft-tokens 4 --spec-draft-margin-tau 1.5

# 3. Counters
curl -s localhost:8000/v1/status | jq .spec_decode
```

A request engages speculative decoding only when `temperature` is `0` (argmax sampling)
and it is the only sequence in the generation batch. Prefix-cache hits disable it for that
request (see *Fallbacks*); benchmark with `--disable-prefix-cache`.

## Status (2026-08-13 to 08-15 measurements, M5 Max 128 GB, Lightning 30B-A3B 6-bit)

| | value |
|---|---|
| live mean accepted draft tokens / round (code generation) | **3.20** of 7 drafted (NVIDIA reports 3.75) |
| per-token acceptance at k=4 | 62 % |
| errors over the measurement runs | 0 |
| plain greedy decode | 115 tok/s whole-request, 134 tok/s steady-state |
| DSpark k=7, first working version (BF16 draft) | 66 tok/s |
| DSpark k=4, tuned (8-bit draft, gathered Markov, sliced head; 86.6 with `--spec-draft-margin-tau 1.5`) | 87–90 tok/s |
| **net** | **0.7–0.8× plain decode** |

Correctness: every emitted token is an argmax of the target's own logits over the emitted
prefix. Against a 1-wide greedy run of the same prompt, 199 positions were compared: 1
divergence, at a position where the 1-wide run's top-1/top-2 logit gap was exactly 0.0000
(a bf16 tie that a different verify width legitimately breaks the other way).

## Clean run on the rebased tree (2026-09-04, host otherwise idle)

Same build for every row; served with `--continuous-batching --disable-prefix-cache --max-num-seqs 1`, native NVFP4 drafter, greedy, thinking off, one warm-up then three 512-token completions per prompt; medians. M5 Max 128 GB, mlx-lm 0.31.3.

| config | code tok/s | vs plain | accepted / round (per token) | prose tok/s | vs plain | accepted / round (per token) |
|---|---|---|---|---|---|---|
| plain greedy | 125.6 | — | — | 118.8 | — | — |
| DSpark k=4 | 104.1 | 0.83× | 2.98 of 4 (74.6 %) | 65.2 | 0.55× | 1.51 of 4 (37.7 %) |
| DSpark k=7 | 76.8 | 0.61× | 3.59 of 7 (51.3 %) | 53.6 | 0.45× | 1.88 of 7 (26.9 %) |
| DSpark k=4, `--spec-draft-margin-tau 1.5` | 109.6 | 0.87× | 2.59 of 2.96 drafted (87.5 %) | 71.1 | 0.60× | 1.17 of 2.22 drafted (52.6 %) |

Counters over the whole run: `errors` 0, `context_disables` 0, `bounded_kv_disables` 0, `despeculations` = one per finished request (the rewind at `extract_cache`). Every config's three runs were byte-identical to each other.

Reading: the ratios match the August measurements (0.7–0.8× then, 0.83–0.87× now on code). Wider blocks lose: k=7 accepts more tokens per round (3.59) but the 8-wide verify costs more than they save. The adaptive cut-off is now the best configuration, because on this tree the cost per drafted position is what it trims. Prose is worse than code at every setting, as the acceptance column predicts.

**Fidelity.** Against the plain run, every DSpark config produced the same two divergences, one per prompt, and all three DSpark configs agree with each other:

| prompt | first divergence | plain chose | DSpark chose | plain-forward logits at that position | gap |
|---|---|---|---|---|---|
| code | token 50 of 512 | `\n\n` | `,` | 28.25 vs 28.25 | **0.000** (exact tie) |
| prose | token 2 of 512 | ` difference` | ` distinction` | 24.125 vs 24.25 | **0.125 = one bf16 ULP** at that magnitude |

The gaps were measured offline with the same weights (the server has no logprobs endpoint): a forward over the prompt plus the common prefix, top-3 logits at the divergence position. In the prose case the wide forward's own argmax is the DSpark token, i.e. the 1-wide run is the one that broke the tie the other way. Both divergences therefore satisfy the acceptance bar proposed in *Greedy versus non-greedy correctness*: every emitted token is an argmax of the target over the emitted prefix, and any difference from a 1-wide run sits at a position whose top-1/top-2 gap is within bf16 resolution.

## Why it does not win yet: the verify-width curve

Speculative decoding pays for a `(k+1)`-wide verify forward instead of `k+1` single-token
forwards. The whole question is how much a wide forward costs. Measured on the same
sequence with a warm ~500-token cache (`mx.eval`-timed, 30 iterations):

| verify width | ms / step | × width-1 | marginal ms / extra token |
|---|---|---|---|
| 1 | 8.47 | 1.00 | — |
| 2 | 13.49 | 1.59 | 5.0 |
| 3 | 15.64 | 1.85 | ~2.2 |
| 5 | 20.67 | 2.44 | ~2.5 |
| 8 | 28.13 | 3.32 | ~2.5 |

Two readings, both true:

* **Sublinear**, so block drafting is the right shape: the dense weights (23 Mamba layers,
  6 attention layers, shared experts, lm_head) are read once per forward regardless of
  width. Only the routed experts scale with width.
* **The marginal cost is the ceiling.** Past width 2 every extra verified token costs
  ~2.5–2.8 ms ≈ 30 % of a full step. That marginal is dominated by per-token routed-expert
  reads (6 of 128 experts × 23 MoE layers ≈ 0.5 GB per token) that the current
  `switch_layers` gather in mlx-lm does not batch across positions. It bounds any
  speculative speedup on this stack at roughly 2×, and reaching 1.5× needs most rounds to
  accept most of the block.

Per-round arithmetic at k=7: 28 ms verify + ~4 ms draft for a mean of ~4.2 kept tokens is
~7.6 ms/token *before* the partial-accept re-advance, which adds another wide forward to
~85 % of rounds. Plain decode is 8.5 ms/token. NVIDIA's numbers work because their fused
batched-expert kernels put the marginal near 1.1×/k.

The ceiling in closed form, with `r` the fraction of rounds that need the re-advance and
`k` the block: `1 / (r + (1 - r) / k)` ≈ 1.7× at the measured acceptance. A realistic
~1.3× on this stack would need a single-forward commit path that hybrid models do not have.

The same curve explains why Lightning's built-in single-token MTP head breaks even here
(74.5 % acceptance and still 106 vs 115–134 tok/s, see #710 for the equivalent measurement on
a hybrid Qwen): with verify-2 at 1.59×, one draft per verify cannot amortize.

**Where the 1.5–2× lives:** batching the routed-expert gather across the verify block would
drop the marginal toward the dense floor (~0.9 ms/token), at which point this exact DSpark
stack projects to ~180–230 tok/s. That kernel work is the natural next phase, and it speeds
up plain MoE decode for every model on vllm-mlx, speculative decoding aside.

## Acceptance is at spec; the residual gap to NVIDIA is the metric

Draft precision was eliminated as a cause: the bit-exact native NVFP4 drafter (MLX
`mode="nvfp4"`) gives identical acceptance to the dequantized BF16→8-bit chain. SPEED-Bench
(real prompts only — 6 of its 11 categories ship as license placeholders), same harness,
same n, only the target precision varying:

| category | 6-bit target | BF16 target | NVIDIA (BF16, temp 1.0 / top_p 0.95) |
|---|---|---|---|
| coding | 4.40 | 4.11 | 4.38 |
| multilingual | 2.13 | 1.96 | 4.55 |
| qa | 2.90 | 3.45 | 3.36 |
| rag | 4.57 | 5.24 | 4.25 |
| writing | 2.04 | 1.92 | 2.83 |
| overall | 3.21 | 3.34 | 3.87 |

Target quantization is a ~4 % effect in mixed directions. coding and rag are at or above
NVIDIA's numbers at both precisions; qa is at the BF16 target. The deficit concentrates in flat-distribution categories (writing,
multilingual) and is largely the metric: NVIDIA's acceptance is probabilistic rejection
sampling at temperature 1.0, ours is exact greedy prefix match, which is strictly harsher
exactly where the distribution is flat.

## Design

### Target model families and artifact format

* **Targets:** `model_type == "nemotron_h"` only (Lightning 30B-A3B; the hybrid
  Mamba-2 + attention + latent-MoE layout). The scheduler hook itself is model-agnostic —
  it needs a drafter that implements the block-draft API below and a target forward that
  can stash aux hidden states.
* **Drafter artifacts**, looked up in the target's model directory:
  1. `dspark-native/model.safetensors` — an MLX-native NVFP4 repack of NVIDIA's checkpoint
     (bit-exact nibbles, `mode="nvfp4"`, group 16, global scales retained). Loaded at
     native precision; `markov_w2` is dequantized exactly to BF16 because it is consumed
     by row gathers.
  2. `dspark/weights.safetensors` — BF16, produced by `scripts/prepare_dspark_weights.py`
     from NVIDIA's NVFP4 checkpoint (fp4-e2m1 nibbles × fp8-e4m3 block scales × fp32 global
     scale). Re-quantized to 8-bit/group-64 at load. DSpark was trained at NVFP4, so 8-bit
     is finer than its native format; measured acceptance is identical to artifact 1.
  * `dspark/config.json` (NVIDIA's config) is always required. The draft vocab (131072) is
    the real tokenizer vocab; the target's 248320 is lm_head padding, so id mapping is the
    identity and the drafter borrows the target's lm_head sliced to 131072 rows.

### How DSpark drafts

* **EAGLE lineage:** the drafter is conditioned on the target's internal hidden states
  after layers `[1, 5, 19, 29, 41, 51]`.
* **DFlash context trick:** the drafter never runs its 6 layers over the context. For each
  token the target processes, the 6 tapped states are concatenated (6×2688), fused by a
  learned `fc` to 2688, RMS-normed, and projected through each draft layer's K/V heads
  directly into that layer's KV cache. Context cost ≈ a few GEMVs per token (~0.6 ms).
* **One parallel block per round:** the query is `[anchor, mask×N]`, the anchor being the
  newest sampled target token and the masks (id 990) sitting at the positions they predict.
  One (1+N)-wide pass through the 6-layer drafter (attention with per-head sink logits and
  a 1024-token sliding window) yields all N predictions at once.
* **Markov fix-up:** parallel drafting cannot see intra-block dependencies, so a rank-512
  head walks left to right adding `markov_w2(markov_w1[prev])` to each position's base
  logits before the argmax. Scored over the top-64 base candidates only (vLLM's gathered
  top-k), so the whole chain stays lazy with one sync per round.

### Draft-block API and fallback behavior

A drafter is any object with:

```text
block_size: int
context_length: int
reset_context() -> None
append_context(aux_cat: [n, num_aux*hidden]) -> None   # positions ctx_len .. ctx_len+n-1
draft(anchor_id: int, target_lm_head, n_draft: int) -> list[int]   # 1..n_draft ids
```

`draft` may return fewer than `n_draft` tokens (adaptive block length:
`--spec-draft-margin-tau` stops at the first position whose Markov-adjusted top-1/top-2
margin falls below tau; every wasted draft position costs a full routed-expert read in
the verify).

Fallbacks, all logged and counted in `/v1/status`:

| condition | behavior |
|---|---|
| no drafter in the model directory | warning at load, server runs plain decode |
| request is not greedy (`temperature != 0`) | generator created without the hook |
| generation batch holds >1 sequence, or logits processors | stock step for that call; hook re-arms when the batch is a single sequence again |
| another request merges into the batch, or a request's cache is extracted (finish/abort), while verified-but-unemitted tokens are in the cache | the cache is rewound to the emitted prefix first (`despeculations += 1`); speculation stays off for that request |
| drafter context ≠ target cache at a cycle (prefix-cache hit, another request's prefill, batch change) | spec off for that request, `context_disables += 1` |
| KV cache no longer trimmable (`--max-kv-size` reached) | spec off for that request, `bounded_kv_disables += 1` |
| any exception in draft/verify | spec off for that request, `errors += 1`; the cache is rewound to the emitted prefix |

### k-wide verify acceptance semantics

The verify forward runs `[P, d1..dk]` in one call. Position `i` of its logits is what the
target emits after `[.., P, d1..di]`. With `v[i] = argmax` at position `i`:

* `m` = number of leading positions with `v[i] == d[i+1]` (a prefix: once a position
  disagrees, nothing after it counts, because it was conditioned on the wrong token).
* `c = v[m]` is the target's own token at the first disagreement, or the bonus token after
  a fully accepted block. It is emitted too, so a round nets `m + 1` tokens and a round
  with `m = 0` still makes progress.

(`vllm_mlx/spec_utils.longest_accepted_prefix`, unit-tested without MLX.)

### KV / recurrent-state rollback

Lightning is 23 Mamba + 23 MoE + 6 attention layers. KV caches can be trimmed to un-process
rejected tokens; recurrent state cannot be rewound. Per round:

1. Snapshot the Mamba layers' state arrays before the verify. MLX cache updates reassign
   the state arrays rather than mutating them, so holding the references *is* a snapshot
   (verified empirically; #710 measures the same kind of snapshot on a hybrid Qwen at
   1.8 % of a step).
2. Verify all `k+1` positions (cache advances by `k+1`).
3. Full accept: nothing to undo. The post-`dk` logits are kept, so the bonus token costs no
   forward.
4. Partial accept: trim the KV caches by `k+1`, restore the Mamba snapshots, and re-advance
   `[P, d1..dm, c]` in one `(m+2)`-wide forward. Both cache types land exactly at the
   emitted sequence. This re-advance is the hard floor a hybrid model imposes: an
   attention-only target could keep the accepted prefix in place and trim only the tail.

### Batch membership and mixed-request rules

The hook lives on `GenerationBatch._step` (mlx-lm ≥ 0.31; the older
`BatchGenerator._step` hook that #737 guards no longer exists). It is single-sequence by
construction: any step where the generation batch holds more than one sequence, or any
logits processor, runs the stock step.

Between two steps of a round the cache is *ahead* of what has been emitted (it holds the
verified-but-unemitted tokens), which stock code must never see. The hook tracks how far
ahead it is and, before a batch merge (`GenerationBatch.extend`), a cache extraction
(`extract_cache`, i.e. finish or abort) or an error fallback, rewinds to the pre-verify
snapshot and re-advances exactly the tokens emitted since. After that the request decodes
plainly for the rest of its life: its drafter context cannot be reconciled with the cache
any more, and the context check (run every cycle, not only the first) fails closed. The
same check catches another request's prefill landing in the aux stash between two steps —
the taps cannot tell sequences apart, so a request that shares the server with a prefill
loses speculation rather than drafting against a foreign context.

Concurrent traffic on one server is therefore correct but not accelerated: the hook only
pays off for stretches where one greedy request has the generator to itself. Generators are
created per sampling params, so non-greedy requests never see the hook; the one pre-existing
gap is upstream's rule that a generator is not recreated while requests are running, so a
non-greedy request inserted into a hooked greedy generator is decoded with the stale sampler
and speculated (same behavior as MTP today; noted for the maintainers). The MLLM path is
untouched.

### Greedy versus non-greedy correctness

Greedy only. The install gate is `temperature == 0`, which `make_sampler` maps to a pure
argmax regardless of `top_p`/`top_k`/`min_p`.

"Byte-identical" is not the right bar, and the definition matters (asked in #699):

* **Guaranteed:** every emitted token id is an argmax of the target's logits over the
  emitted prefix. This is checkable after the fact by re-scoring the emitted sequence in a
  plain forward.
* **Not guaranteed, and not a bug:** identical token ids to a separate 1-wide greedy run.
  bf16 logits computed at width 8 and width 1 differ in reduction order; where the target's
  own top-1/top-2 gap is within that noise, the two runs may break the tie differently.
  Measured: 1 flip in 199 positions, at a gap of exactly 0.0000.
* Decoded bytes follow from token ids, so the same statements hold for them.

Proposed acceptance test: (a) re-scored argmax agreement at 100 % of emitted positions,
and (b) any divergence from the 1-wide run must sit at a position whose 1-wide top-1/top-2
gap is ≤ bf16 epsilon of the logit magnitude. Non-greedy (rejection-sampling) verification
is a follow-up; the verify forward already produces the full distributions it needs.

### Metrics

`GET /v1/status` → `spec_decode`:

| field | meaning |
|---|---|
| `rounds` | draft/verify rounds attempted |
| `drafted` / `accepted` | draft tokens proposed / accepted |
| `acceptance_rate` | `accepted / drafted` (per-token) |
| `mean_accept_len` | `accepted / rounds`; a round also emits the target's own token after the accepted prefix, so tokens per round is this + 1 after a full block and this + 2 after a partial one (the correction plus the next sample) |
| `rounds_full` | rounds where the entire block was accepted |
| `context_disables` | requests where spec turned itself off at the context check |
| `bounded_kv_disables` | requests where spec turned itself off because the KV cache was no longer trimmable |
| `despeculations` | times the cache was rewound to the emitted prefix ahead of a batch merge or extraction |
| `errors` | requests where spec turned itself off on an exception |
| `active` | whether the current request still has spec enabled |

Counters are cumulative across generator replacements (same pattern as the MTP counters).

### Benchmark matrix

What a throughput or output-fidelity claim for this path needs to state:

| dimension | values |
|---|---|
| target | Lightning 30B-A3B 6-bit (primary); BF16 (precision control) |
| drafter artifact | native NVFP4; BF16→8-bit (must agree) |
| `k` (`--spec-num-draft-tokens`) | 1, 2, 3, 4, 5, 7 |
| `--spec-draft-margin-tau` | off, 1.0, 1.5, 2.0 |
| prompts | SPEED-Bench real categories (coding, multilingual, qa, rag, writing) + one fixed 512-token code prompt for steady-state timing |
| regime | single stream, `temperature 0`, `--disable-prefix-cache`, warm cache (report the depth) |
| throughput | whole-request tok/s and steady-state decode tok/s, each vs plain greedy on the same server build |
| acceptance | `mean_accept_len`, `acceptance_rate`, `rounds_full` |
| fidelity | re-scored argmax agreement; divergence count vs 1-wide run with the 1-wide top-1/top-2 gap at each divergence |
| verify-width curve | ms/step at widths 1, 2, 3, 5, 8 on the same cache depth (the number that predicts everything else) |
| environment | chip / memory, `mlx`, `mlx-lm`, vllm-mlx tree |

The measurements on this page: M5 Max 128 GB, mlx-lm 0.31, vllm-mlx at the #699 spike
tree (fork branch, June 2026 base), 2026-08-13 to 08-15.

## Limitations

* Greedy, single-sequence, NemotronH targets only.
* Prefix-cache hits disable speculation for that request (no aux states exist for cached
  tokens). Re-deriving them from a cached prefix is possible but not implemented.
* A stop or `max_tokens` inside a block wastes the already-verified trailing tokens (the
  cache handed back at finish is rewound to the emitted prefix, so nothing leaks into the
  prefix cache).
* Not profitable on current MLX kernels; see the width curve. Ship-worthy once the routed
  expert gather batches across the verify block.

## Bugs worth knowing about (each cost real hours)

* **fp8-e4m3 block scales read as integers.** NVFP4 block scales are e4m3 *bit patterns*
  in a uint8 tensor; `astype(float32)` silently yields 0–255. Teacher-forced agreement was
  15.6 % with the bug and 29.9 % after decoding the pattern. If a converted checkpoint
  "mostly works", check the scales first (`vllm_mlx/spec_utils.fp8_e4m3_to_float`).
* **A BF16 drafter defeats itself.** 967M parameters in BF16 is 1.9 GB; a draft round then
  reads as many bytes as a target step (~8 ms measured). 8-bit quantization takes the
  round from 6.1 to ~4 ms with identical agreement, because the drafter was trained at
  4-bit.
* **Full-vocab Markov per step.** The naive sequential loop reads a 131k×512 matrix and
  syncs the GPU 7 times per round. Scoring only the top-64 base candidates makes the whole
  chain lazy with one sync per round.
* **Clearing the aux stash on a new request** starved the context check and disabled
  speculation for every request after the first: at that moment the stash already holds
  the new request's prompt states.
