#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${VLLM_MLX_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
PORT="${PORT:-8000}"
LISTEN_MODE="${LISTEN_MODE:-localhost}"

usage() {
  cat <<'EOF'
Usage:
  scripts/serve_profile.sh <profile> <model> [extra serve args...]

Profiles:
  text-default         Simple text profile for local usage
  text-deterministic   Reproducible text diagnostics profile
  text-tools           Text tool-calling profile
  text-json            Long-timeout JSON/extraction profile
  mllm-default         Simple multimodal profile
  mllm-correctness     Multimodal profile with divergence monitoring + serialize

Environment:
  VLLM_MLX_PYTHON  Override python entrypoint (default: .venv/bin/python)
  PORT             Override serve port (default: 8000)
  LISTEN_MODE      localhost or public (default: localhost)

Examples:
  scripts/serve_profile.sh text-default mlx-community/Qwen3-4B-Instruct-2507-4bit
  scripts/serve_profile.sh text-json mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
  scripts/serve_profile.sh mllm-default mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

PROFILE="$1"
MODEL="$2"
shift 2

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python entrypoint not found or not executable: ${PYTHON_BIN}" >&2
  echo "Create the checkout venv first, or set VLLM_MLX_PYTHON." >&2
  exit 1
fi

LISTEN_ARGS=()
case "${LISTEN_MODE}" in
  localhost)
    LISTEN_ARGS+=(--localhost)
    ;;
  public)
    LISTEN_ARGS+=(--host 0.0.0.0)
    ;;
  *)
    echo "Unsupported LISTEN_MODE: ${LISTEN_MODE}" >&2
    echo "Use LISTEN_MODE=localhost or LISTEN_MODE=public." >&2
    exit 1
    ;;
esac

BASE_ARGS=(
  serve
  "${MODEL}"
  "${LISTEN_ARGS[@]}"
  --port "${PORT}"
)

PROFILE_ARGS=()
case "${PROFILE}" in
  text-default)
    PROFILE_ARGS+=(
      --runtime-mode simple
      --cache-strategy auto
      --default-temperature 0.7
      --default-top-p 0.9
    )
    ;;
  text-deterministic)
    PROFILE_ARGS+=(
      --deterministic
    )
    ;;
  text-tools)
    PROFILE_ARGS+=(
      --runtime-mode auto
      --cache-strategy auto
      --enable-auto-tool-choice
      --tool-call-parser auto
    )
    ;;
  text-json)
    PROFILE_ARGS+=(
      --runtime-mode simple
      --cache-strategy auto
      --timeout 900
      --default-temperature 0.7
      --default-top-p 0.9
    )
    ;;
  mllm-default)
    PROFILE_ARGS+=(
      --runtime-mode simple
      --cache-strategy auto
      --mllm
      --default-temperature 0.0
      --default-top-p 1.0
    )
    ;;
  mllm-correctness)
    PROFILE_ARGS+=(
      --runtime-mode auto
      --cache-strategy auto
      --mllm
      --default-temperature 0.0
      --default-top-p 1.0
      --batch-divergence-monitor
      --batch-divergence-threshold 0.95
      --batch-divergence-action serialize
    )
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    usage
    exit 1
    ;;
esac

CMD=(
  "${PYTHON_BIN}"
  -m
  vllm_mlx.cli
  "${BASE_ARGS[@]}"
  "${PROFILE_ARGS[@]}"
  "$@"
)

echo "Launching profile '${PROFILE}' for model '${MODEL}'"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
