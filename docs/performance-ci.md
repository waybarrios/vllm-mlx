# Apple Silicon performance CI

The **Apple Silicon performance** workflow compares relevant PRs with their
base commit on a standard `macos-15` Apple Silicon runner. Nightly and manual
runs compare the selected branch commit with its first parent. Standard hosted
runner compute is free for public repositories; artifact storage has separate
allowances. This workflow retains small JSON/log artifacts for seven days and
does not upload models or use larger paid runners.

Each job downloads immutable model snapshots outside the timed region and
installs one shared runtime. `scripts/performance_ci/workloads.json` records
the model revisions, critical runtime pins and workload parameters. Keep its
runtime pins synchronized with `requirements.txt`. Other dependencies resolve
once per job and their installed versions are recorded. This measures source
changes under a fixed runtime, not dependency upgrades or installation speed.

PRs use the quick profile: Qwen3 0.6B 8-bit, five paired repeats, three cases:

| Case | Prompt tokens per request | Output tokens per request | Concurrent requests | Prefix cache |
|---|---:|---:|---:|---|
| Decode | 128 | 64 | 1 | Off |
| Batch | 512 | 64 | 4 | Off |
| Prefix reuse | 512 | 64 | 1 | On |

Nightly adds Llama 3.2 1B 4-bit and uses seven repeats. Manual dispatch offers
either profile. Each pair runs sequential fresh processes in alternating
base/candidate order, on the same VM. Model loading, generation, synchronization
and cleanup use one model-owning thread per process.

Each process records model load time, the first batch and a second batch with
identical prompts. First batch means a freshly loaded model and process; it
does not imply cold filesystem caches. Warm measurements include compilation
reuse and, only in the prefix case, prompt-cache reuse. Cache-hit tokens are
recorded and zero warmed prefix hits produce a warning.

Prompt IDs have exact lengths and distinct request suffixes. Greedy sampling
and a benchmark-only EOS logits mask enforce fixed output lengths. Batch
throughput includes prefill and streaming through terminal completion. TTFT is
the median request time from submission to the first streamed token, including
queue/admission time. Peak MLX bytes include model weights and active caches;
this is not process RSS or total system memory. Model load is reported separately.

The job summary reports medians, median absolute deviations and the median
of paired candidate/base percentage changes. Higher throughput is better;
lower latency and memory are better. Inspect raw samples and their spread
before interpreting changes. Output digests that differ produce visible
warnings; this performance workflow does not establish generation correctness.

Performance changes are initially **report only**. Collect runner variability
before choosing thresholds or making the job a required performance gate.
Crashes, timeouts, incomplete pairs, non-finite metrics, wrong token counts or
incompatible prompts/runtime/source provenance fail the job. Existing functional
tests and throughput acceptance thresholds remain in force.

Download the `apple-silicon-performance-*` artifact for individual JSON records,
worker logs, model file hashes, source SHAs, runtime versions and hardware
provenance. Workers have 180-second process-group deadlines; jobs have a
60-minute bound. The workflow uses `pull_request` with read-only permissions,
no persistent checkout credentials, no secrets and no PR comments.

To reproduce on an Apple Silicon Mac with two clean source checkouts and the
shared runtime installed:

```sh
python scripts/performance_ci/controller.py \
  --base ./baseline --head ./candidate --profile quick \
  --output ./performance-results
```

The output directory must not already exist. Both source paths must refer to
clean committed checkouts. Run the harness from the candidate source tree.
