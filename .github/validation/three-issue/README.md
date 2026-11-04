# Final native acceptance operator bundle

Source replay only has been verified. Full configured-suite acceptance and
physical Metal/model measurements remain **PENDING**. No native run, transfer,
install, download, commit, CI trigger, or publication was performed here.

`manifest.json` binds pinned base `3af7d7d61576606153f419816be2f4a1a9baab06`,
the frozen common v2 layer, and three separate narrow issue layers. Source
manifests include every tracked file and each untracked payload/mode. `SHA256SUMS`
checks every bundled input. `historical-original/` is the unchanged earlier
native protocol and harness, retained as provenance; execute the final scripts
at this bundle's root and `scripts/`, not the historical candidate patch.

## Prepare source archives

Use an existing repository containing the pinned commit and one already
provisioned isolated Mac interpreter. All run output and writable cache state
stay beneath that repository; existing model caches and snapshots may be
external read inputs. Supply absolute paths; do not reuse an existing run root.

```bash
export REPO="/absolute/path/to/existing/vllm-mlx"
export BUNDLE="$REPO/.scratch/goal-three-fixes-20261101/native-final-acceptance"
export PYTHON="/absolute/path/to/provisioned/isolated/bin/python"
export RUN_ROOT="$REPO/.scratch/native-final-run"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" "$BUNDLE/replay_sources.py" --repo "$REPO" --run-root "$RUN_ROOT"
```

This archives the exact pinned commit five times, checks and applies the common
patch plus one issue patch, restores untracked modes, and verifies every source
hash. Baseline production remains exactly the pinned archive. The identical
final benchmark is copied into `baseline/scripts/benchmark_cache_release.py`;
its own `sys.path` insertion therefore imports baseline. `replay-proof.json`
records apply checks and the baseline import root. These are manifest-bound
archives without `.git`: **never infer their identity with `git rev-parse`**,
which could report the parent repository's unrelated checkout.

## Bind operator and model identities

Copy `operator.example.json` to a project-local operator file and replace every
placeholder. Required fields identify the interpreter, replay root, existing
HF Hub cache (which may be outside the project), and exact cached `refs/main`
revisions for:

- `mlx-community/Llama-3.2-1B-Instruct-4bit`: deterministic batching and native
  EngineCore stream fixtures.
- `mlx-community/Qwen3-0.6B-8bit`: continuous batching and detokenizer fixtures.

Preflight requires both unchanged model IDs, exact revision refs, complete
tokenizer/weight snapshots, and content-addressed blob checks. It records every
cached artifact SHA256 with offline mode enabled. A benchmark `--model` path
never replaces these tests' fixed model IDs. Do not set `VLLM_MLX_TEST_MODEL`,
inject `PYTEST_ADDOPTS`, change model IDs, or modify fixture paths.
The supplied Hub cache is read input only; the runner does not copy/download
models or modify it. `HF_HOME`, `HF_ASSETS_CACHE`, `HF_MODULES_CACHE`, dataset
cache, temporary files and other writable cache state point under run output.

`fixture-models.json` also records slow/default-deselected model IDs and the
unchanged local/opt-in MTP paths. Preserve existing markers and flags; report
their skips truthfully. Default pytest remains `not slow and not integration`.
Configured-suite success alone does not establish those excluded cases.

For measurement mode, provide one vetted immutable single-model directory,
its revision, and an existing registry YAML selecting at least two local
continuous-batching residents. Keep the same artifacts for baseline/candidate.
Create complete local weight manifests with this read-only inventory command
for the single model and each registry resident:

```bash
"$PYTHON" "$BUNDLE/inventory_model.py" --repo "$REPO" \
  --model "/absolute/path/to/existing/model-snapshot" \
  --output "$REPO/.scratch/native-inputs/single-model-files.json"
```

Fill `benchmark_weights_manifest` and each `registry_artifacts` entry with those
files and immutable revision labels. Preflight checks every declared model
file, the unchanged YAML paths, and records dependency/runtime versions and
Metal device information. No installation or fetching is included.

## Execute and retain terminal results

Run locally only on actual Darwin arm64 with `mx.metal.is_available()`. All
commands below create separate UTC-labelled results directories; none runs here.

```bash
export OPERATOR="$REPO/.scratch/native-inputs/operator.json"
"$PYTHON" "$BUNDLE/run_native.py" --operator "$OPERATOR" --phase suites
"$PYTHON" "$BUNDLE/run_native.py" --operator "$OPERATOR" --phase focused
"$PYTHON" "$BUNDLE/run_native.py" --operator "$OPERATOR" --phase measurements
# GPU acceptance uses an explicit policy and still reports strict RSS results.
"$PYTHON" "$BUNDLE/run_native.py" --operator "$OPERATOR" --phase measurements --memory-accounting gpu-footprint
```

The full configured suite is the historical unchanged
`python -m pytest -q --tb=short -p no:cacheprovider` on pinned baseline, common,
and all three candidates, with **600 seconds then TERM and 10 seconds before
KILL**. The earlier outer 120-second budget timed out even on the pinned baseline;
those failed runs remain evidence. Focused processes retain their 120-second
budget. Existing pytest markers, models, per-request deadlines, and assertions
are untouched. Exit 124, collection failure, missing terminal summary, and an
unexpected fixture skip are not successful acceptance. Focused issue commands
supplement this full gate; the original three #793 native commands remain.

Measurement profiles preserve the original synthetic/production × prefix/all
× three fresh-process repetitions × baseline/candidate matrix:

- `historical24`: the original 24 processes, with `owner-streams`, unchanged;
  at least 72 minutes of settling.
- `acceptance48` (default): the original 24 plus production-registry runs under
  both `owner-streams` and `none`, another 24; at least 144 minutes of settling.
- `exhaustive96`: optional diagnostic coverage of all four modes and both
  barriers; at least 288 minutes of settling. It is not a required acceptance
  gate.

Select explicitly with `--measurement-profile historical24`,
`--measurement-profile acceptance48`, or `--measurement-profile exhaustive96`
on the measurements command. Results record the selected profile. Each process
keeps three cycles, 256MiB KV, 1024MiB cache cap, 60-second settling, three
requests and 1024 prompt repeats. Measurement processes have the original
unbounded outer window because three settling cycles already exceed 120s.
Model load/generation adds to those minima; do not fit a profile into a
30-minute job by reducing work.

Both accounting policies require at least 16MiB attributed workload,
populated then cleared entries per resident, at least 80% active and
active-plus-cache reduction, immediate unused-pool zero, at least 50%
physical-footprint reduction and a footprint within 32MiB of resident baseline.
The default `strict-rss` policy additionally requires the historical 50% RSS
reduction and 32MiB RSS plateau. The `gpu-footprint` policy retains both RSS
checks as diagnostics because the hosted samples show Metal allocations in
physical footprint while RSS is insensitive. It cannot be reported as strict
RSS success: each summary separately records `strict_rss_outcome` and its exact
failed predicates. Policy version2 and the selected policy are explicit in
the logs; the controller rejects mixed policies or missing diagnostics.

The hosted controller explicitly selects `gpu-footprint` for both baseline and
candidate, with all 48 processes and their original workload intact. Historical
strict runs and their failed assertions are retained separately. Retain
resident identity, unchanged idle step count, and successful
post-clear generation. No post-DELETE test-side GC, clear, synchronization,
new request or unload rescues a result. The no-barrier variant must remain
separate from synchronized allocator attribution.

Owner-stream barriers synchronize each resident's default stream and its
captured producing batch stream on that resident's worker. Process-global
generation-stream handles can belong to another worker and are not valid
substitutes. The completed historical strict matrix exposed this cross-worker
failure; its failed processes remain part of the retained evidence.

A successful HTTP response with an unsupported DELETE status is recorded as
a failed cache-clearing predicate through all cycles, rather than aborting
before measurement. HTTP errors still fail the process. Baseline failures stay
failures, including when reported entry counts happen to be zero.

Each run saves exact argv, cwd, source/bundle identities, bounded timeout and
exit code, raw stdout/JSONL and stderr, model/cache manifests and versions.
Expected baseline failures do not abort the matrix; they remain nonzero in the
results. Inspect all candidate terminal summaries and every repetition, report
failed gates and variance, and retain synthetic/production labels. This bundle
establishes no TTFT, throughput, output-quality, or physical-memory success.
