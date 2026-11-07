# Apple Silicon performance CI Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Detect performance changes with repeated, comparable native measurements.

**Architecture:** A portable controller launches sequential isolated workers for
two source checkouts. The native worker exercises EngineCore on one model-owning
thread. GitHub Actions supplies immutable source refs and retains evidence.

**Tech Stack:** Python 3.12, pytest, MLX, GitHub Actions macos-15.

**Spec:** `docs/superpowers/specs/2026-11-05-apple-silicon-performance-ci-design.md`

## Global constraints

- Public standard `macos-15`, read-only permissions, immutable action/model refs.
- Five quick paired repeats, seven nightly paired repeats; no performance gate.
- Preserve existing test assertions; register new test files in CI.
- No commit, push or PR creation without explicit approval.

## Review focus

- Missing or duplicate observations must fail rather than bias the report.
- Non-finite/zero metrics and mismatched prompts/runtime must fail.
- Hung children must terminate with retained logs and failed run status.
- Early EOS must not shorten workloads; cache-reuse results must expose hits.
- Fork PR execution must have no credentials, secrets, writes or comments.

### Task 1: Paired controller and report

**Files:** `scripts/performance_ci/{__init__.py,controller.py,workloads.json,requirements.txt}`,
`tests/test_performance_ci.py`.

**Interfaces:** `pair_order(repeat)`, `summarize(records, cases, repeats)`,
`run_child(command, log_path, timeout, env)`; CLI accepts source paths, profile,
output directory and summary destination.

- [x] Write and observe failing tests for alternating order, complete pairing,
  invalid metrics, incompatible provenance, paired percentage arithmetic,
  and real subprocess failure/timeout.
- [x] Implement the controller, fixed cases/runtime, JSON and Markdown reports.
- [x] Run the portable tests and inspect CLI help.

### Task 2: Native fixed-workload worker

**Files:** `scripts/performance_ci/worker.py`, `tests/test_performance_ci_native.py`.

**Interfaces:** `make_prompts(tokenizer, length, concurrency)`,
`eos_processor(token_ids)`, `measure(loader, case)`; CLI validates source and GPU.

- [x] Write and observe failing tests for exact distinct prompts and real tiny
  MLX generation, fixed output length, cold/warm metrics and prefix hits.
- [x] Implement one owner thread for loading, generation, synchronization and
  cleanup; fail on missing terminal results, wrong counts and timeouts.
- [x] Run the native smoke using a tiny in-memory model, with no model downloads.

### Task 3: Workflow, documentation and verification

**Files:** `.github/workflows/{performance.yml,ci.yml}`, `docs/performance-ci.md`.

- [x] Wire relevant PR paths, nightly and manual profiles; immutable source
  checkout, fixed dependencies, bounded runs, always-upload logs and summary.
- [x] Register new tests and migrate the retired macos-14 label to macos-15.
- [x] Run scoped and repository lint/format/type checks, workflow validation,
  portable CI tests and native smoke. Record limitations precisely.
- [x] Obtain fresh correctness and security review, address verified findings,
  then present exact commit message and files for approval.
