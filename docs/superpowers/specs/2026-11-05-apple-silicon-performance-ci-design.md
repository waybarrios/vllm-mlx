# Apple Silicon performance CI

Use the standard public-repository `macos-15` runner. Run a report-only
comparison for relevant PRs, nightly, and on manual dispatch. Existing test
thresholds remain unchanged. A performance difference alone never fails this
first version; missing or invalid measurements do.

Compare the PR base SHA with its head SHA, or the default branch with its first
parent for scheduled/manual runs. Use one Python environment and pinned MLX,
mlx-lm, mlx-vlm, transformers and immutable model revisions. Dependency changes
are outside this fixed-runtime comparison. Download models before timing.

Run sequential fresh processes in alternating base/head order, five paired
repeats per case for quick runs and seven for nightly runs. Quick uses Qwen3
0.6B 8-bit; nightly also uses Llama 3.2 1B 4-bit. Cases cover single decode,
batched prefill/decode, and prefix reuse. Fix prompt/output token counts and
greedy sampling; a benchmark-only EOS logits mask enforces output length.

Measure model load separately, then the first batch and a repeated warmed
batch. First-batch means a fresh model process, not cold disk/OS caches.
Report batch throughput, median request TTFT, batch latency and peak MLX
allocated bytes. Preserve prompt/output digests, cache-hit tokens, raw samples,
source SHAs, runtime versions and hardware information. Output differences
are visible warnings, not evidence of correctness or equivalent generations.

Summaries show base/head medians, median absolute deviation and median paired
relative changes. No threshold is chosen before collecting runner variability.
Subprocesses have deadlines and kill their process groups on expiry. Archive
JSON/logs for seven days, never model weights. Keep read-only permissions,
avoid secrets and PR comments, pin actions, and use `pull_request`, never
`pull_request_target`.
