# Issue 793 native acceptance recovery bundle

Prepared 2026-11-01 from the unchanged, previously reviewed candidate. **Instructions only: no Apple execution, transfer, install, workflow trigger or publication has occurred.** Run only on a permitted real Apple Silicon Mac with a vetted, already-present model and provisioned isolated Python environment. Synthetic results do not establish production-model acceptance.

## Recovered identity

- Base: `3af7d7d61576606153f419816be2f4a1a9baab06`.
- Candidate is an uncommitted seven-path delta at that same HEAD, not a published commit. Identify it by `candidate-793.patch` SHA-256 `e9166bb24a1197f7fb50c4a79150208c53f918adfc4e6182a1acebc4c065bad8` and the seven hashes in `candidate-source-SHA256SUMS`.
- Exact corrected harness: `scripts/benchmark_cache_release.py`, 374 lines, SHA-256 `c325d38bc026b66c61fc2d4fe78906a6589e4c9c310fad3bb0929e9ada56d20b`. Terminal `finish_reason=error` raises; valid immediate EOS remains valid. Candidate includes the real `AsyncEngineCore.clear_prefix_cache()` delegate.
- `bundle-SHA256SUMS` checks the recovered harness, patch and source-hash manifest. README is documentation, not an executable wrapper.
- Current Linux reference environment only: Python 3.12.3, mlx/mlx-cpu 0.32.4, mlx-lm 0.31.3, mlx-vlm absent, httpx 0.28.1, FastAPI 0.162.0, pytest 9.2.0. **These are not Mac measurements or a complete Mac lockfile.** Use one identical, already-provisioned Mac interpreter for base and candidate, record its full installed versions, and never install the CPU backend as a Metal substitute.
- Surviving dependency contract: project `mlx>=0.29.0`; benchmark checks actual `mx.metal.is_available()`. Do not spoof platform/Metal predicates.

## Required user/operator inputs

1. Bring an existing macOS host online; explicitly authorize its SSH alias/account or execute locally yourself. Confirm Darwin arm64 and actual Metal, not just a device name.
2. Supply an existing local repository containing the pinned base commit and a provisioned isolated Mac interpreter with this project's dev/API/vision dependencies. No shared-environment installation commands are included.
3. Supply a vetted immutable local model snapshot, its exact revision/digest, model/quantization identity, and enough free memory. No implicit model downloads are allowed; the harness forces HF offline flags and disables remote-code trust. This is not a sandbox for hostile snapshots.
4. Authorize how this bundle reaches that Mac; **no transfer/upload is authorized or performed by this document**. Keep results local until a separate sharing decision.

No git commit/push/PR, GitHub workflow dispatch or paid compute is necessary for the local procedure below. Those actions remain separately approval-gated.

## A. Read-only capability probe, only after host authorization

For a user-supplied existing alias, not a guessed hostname:

```bash
ssh "$APPROVED_MAC_ALIAS" 'uname -sm; sw_vers; python3 -c "import platform; print(platform.platform(), platform.machine())"'
```

Then run locally on the Mac with paths explicitly set by the operator:

```bash
export REPO="/absolute/path/to/existing/vllm-mlx"
export BUNDLE="/absolute/path/to/native-acceptance-bundle"
export MAC_ROOT="/absolute/path/to/disposable-793-acceptance"
export PYTHON="/absolute/path/to/provisioned/isolated/bin/python"
export MODEL_PATH="/absolute/path/to/existing/immutable/model-snapshot"
export MODEL_REVISION="immutable-revision-or-weight-manifest-digest"
export BASE_SHA=3af7d7d61576606153f419816be2f4a1a9baab06
export BASE_CHECKOUT="$MAC_ROOT/base"
export CANDIDATE_CHECKOUT="$MAC_ROOT/candidate"
export RESULTS="$MAC_ROOT/results"
test -x "$PYTHON" && test -f "$MODEL_PATH/config.json"
git -C "$REPO" cat-file -e "$BASE_SHA^{commit}"
(cd "$BUNDLE" && shasum -a 256 -c bundle-SHA256SUMS)
"$PYTHON" - <<'PY'
import platform
assert platform.system() == "Darwin" and platform.machine() == "arm64"
import mlx.core as mx
assert mx.metal.is_available(), "Real Metal required"
print(platform.platform(), platform.machine())
print(getattr(mx, "device_info", mx.metal.device_info)())
PY
```

Stop if these fail. Do not fall back to Linux/CPU or download a replacement model.

## B. Disposable source preparation, local only

These commands are operator preparation after the inputs/permission above exist; they have **not** run in the current session. They create local detached checkouts without commits or changing any existing reviewed branch. Do not reuse nonempty paths.

```bash
mkdir -p "$MAC_ROOT" "$RESULTS"
git -C "$REPO" worktree add --detach "$BASE_CHECKOUT" "$BASE_SHA"
git -C "$REPO" worktree add --detach "$CANDIDATE_CHECKOUT" "$BASE_SHA"
git -C "$CANDIDATE_CHECKOUT" apply --check "$BUNDLE/candidate-793.patch"
git -C "$CANDIDATE_CHECKOUT" apply "$BUNDLE/candidate-793.patch"
(cd "$CANDIDATE_CHECKOUT" && shasum -a 256 -c "$BUNDLE/candidate-source-SHA256SUMS")
# Base has no benchmark. Copy identical measurement code ONLY; do not fix base.
cp "$BUNDLE/scripts/benchmark_cache_release.py" "$BASE_CHECKOUT/scripts/benchmark_cache_release.py"
cmp "$BASE_CHECKOUT/scripts/benchmark_cache_release.py" "$CANDIDATE_CHECKOUT/scripts/benchmark_cache_release.py"
git -C "$BASE_CHECKOUT" diff --exit-code
git -C "$CANDIDATE_CHECKOUT" diff --check
```

The base benchmark is intentionally an untracked harness addition; base production/test files stay unchanged. Its text-prefix route lacks the wrapper delegate, and its DELETE routes lack pool finalization. Preserve the resulting failures rather than adding test-side cleanup to make base pass. The candidate must match all recovered source hashes; HEAD equality alone is insufficient.

## C. Environment/model manifest and focused native tests

```bash
{
  date -u
  sw_vers
  /usr/sbin/sysctl -n machdep.cpu.brand_string hw.memsize
  "$PYTHON" --version
  printf 'MODEL_REVISION=%s\n' "$MODEL_REVISION"
  shasum -a 256 "$MODEL_PATH/config.json"
  git -C "$BASE_CHECKOUT" rev-parse HEAD
  git -C "$CANDIDATE_CHECKOUT" rev-parse HEAD
  shasum -a 256 "$BUNDLE/candidate-793.patch"
} > "$RESULTS/environment.txt"
"$PYTHON" -m pip freeze --all > "$RESULTS/installed-versions.txt"
"$PYTHON" - <<'PY' > "$RESULTS/metal-device.json"
import json, mlx.core as mx
assert mx.metal.is_available()
print(json.dumps(getattr(mx, "device_info", mx.metal.device_info)()))
PY
(
  cd "$CANDIDATE_CHECKOUT"
  "$PYTHON" -m pytest tests/test_server_cache_controls.py tests/test_engine_core_request_owner.py -q --tb=short
) > "$RESULTS/candidate-portable-contracts.log" 2>&1
(
  cd "$CANDIDATE_CHECKOUT"
  "$PYTHON" -m pytest tests/test_mllm_scheduler_stop_release.py -q --tb=short
) > "$RESULTS/candidate-shutdown.log" 2>&1
(
  cd "$CANDIDATE_CHECKOUT"
  "$PYTHON" -m pytest tests/test_memory_cache_mlx.py -q --tb=short
) > "$RESULTS/candidate-native-tests.log" 2>&1
```

Review exit codes immediately; a skip is not a native pass. Do not output serial numbers, environment dumps, credentials or raw generated text. OS build and MLX/device versions identify the tested Metal stack; no separate unobserved driver-version number should be invented. Record the model name and quantization separately from these manifests; config hash alone is not a weight digest.

## D. Fixed measurement commands, declared before any run

No thresholds or code changes between revisions/repeats. Each fresh process runs **three fill/clear/settle cycles**, keeping the same model resident. Four measurement variants are base/candidate × prefix/all; each has three fresh-process repetitions. Run synthetic and production modes separately; label them accurately. Fixed flags below match the surviving script; `--cache-limit-mib 1024` deliberately chooses a controlled high retained-pool target, **not a default-policy performance certification**.

```bash
set +e  # Preserve expected baseline failures; every exit code is recorded below.
printf 'revision\tmode\troute\trepeat\texit_code\n' > "$RESULTS/exit-codes.tsv"
for mode in synthetic production; do
  for route in prefix all; do
    for repeat in 1 2 3; do
      for revision in base candidate; do
        if [ "$revision" = base ]; then checkout="$BASE_CHECKOUT"; else checkout="$CANDIDATE_CHECKOUT"; fi
        stem="$RESULTS/$revision-$mode-$route-repeat-$repeat"
        if [ "$mode" = production ]; then
          (
            cd "$checkout"
            PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/benchmark_cache_release.py \
              --model "$MODEL_PATH" --route "$route" --cycles 3 \
              --kv-mib 256 --cache-limit-mib 1024 --settle-seconds 60 \
              --requests 3 --prompt-repeats 1024
          ) > "$stem.jsonl" 2> "$stem.stderr"
        else
          (
            cd "$checkout"
            PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/benchmark_cache_release.py \
              --route "$route" --cycles 3 --kv-mib 256 \
              --cache-limit-mib 1024 --settle-seconds 60 \
              --requests 3 --prompt-repeats 1024
          ) > "$stem.jsonl" 2> "$stem.stderr"
        fi
        rc=$?
        printf '%s\t%s\t%s\t%s\t%s\n' "$revision" "$mode" "$route" "$repeat" "$rc" >> "$RESULTS/exit-codes.tsv"
      done
    done
  done
done
```

Run the measurement loop with shell `errexit` disabled so expected base failures are retained and the matrix continues. This does **not** convert nonzero results to passes: inspect every recorded exit code and require each candidate run to emit a successful summary. No remote listener, CI trigger or external result upload occurs. Twenty-four processes × three 60-second settling windows implies **at least 72 minutes** for the full synthetic+production matrix, plus load/generation; production-only is at least 36 minutes. Do not promise completion inside the existing client-acceptance workflow's 30-minute timeout.

## E. Unchanged gates and interpretation

For each cycle, preserve actual samples and compute the trajectory, not just a peak:

- Samples: `before_load`, `after_load`, `resident_baseline`, `filled_idle`, and after clear at 0/1/5/10/60 seconds. Current RSS = `proc_pid_rusage` resident bytes; current physical footprint = its `phys_footprint`; MLX active, cached, and sum are separate fields.
- Workload must be >=16MiB attributed bytes, with populated cache entries before and zero entries after.
- Active and active+cache drops each >=80% of attributed bytes; immediate unused pool zero.
- Current RSS and footprint drops each >=50%; settled RSS/footprint each <=resident baseline +32MiB.
- Model identity remains resident, step count unchanged during idle, post-clear real generation succeeds; terminal errors cannot earn a passing summary. Immediate EOS alone is valid.
- No sampling-window GC, synchronization, allocator drain, new request, unload, or test-side clear. The one setup drain occurs before first fill. Producer streams are synchronized on the owner before the filled-idle sample; second DELETE only follows the settled sample; stop only follows all measurement results.
- Synthetic bytes are explicitly 256MiB KV +16MiB persistent owner. Production attribution is the measured active delta over resident baseline, not proof every byte is KV. Driver/runtime initialization may fail the strict plateau; retain/investigate it, never loosen gates after observing results.
- Report base and candidate per route/model/mode, all repetitions and failed gates, plus median settled deltas and variance. Missing summary, exception, skipped native case, refusal exit 2, insufficient workload, or a synthetic-only success is not production acceptance. No TTFT/ITL/throughput/quality claim is justified by this harness.

No current Apple results exist. If the exact workload fails, return raw logs and the observed bottleneck/ownership/stream evidence for a separately scoped decision; do not edit the reviewed candidate here.

## Existing CI is not a substitute for this packet

`ci.yml` has a hosted `macos-14` Apple job, and both cache-control/native memory files are listed, but runs only for main pushes or PRs targeting main. No candidate publication or trigger is authorized. Native tests are synthetic; the workflow does not supply a real local model for the production matrix or retain its raw JSONL. `client-acceptance.yml` requires a registered self-hosted macOS ARM64 runner, a prepared server and explicit dispatch, and is main-only; current runner API count is zero. Neither historical green CI nor a future synthetic green result establishes the missing production RSS/footprint gate.
