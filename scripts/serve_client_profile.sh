#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVE_PROFILE_SCRIPT="${REPO_ROOT}/scripts/serve_profile.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/serve_client_profile.sh <client-profile> <model> [extra serve args...]

Client profiles:
  goose-text          Goose CLI basic text validation
  goose-tools         Goose CLI with tool-calling enabled
  open-webui-text     Open WebUI text connection
  open-webui-mllm     Open WebUI multimodal connection
  cherry-studio       Cherry Studio desktop text baseline
  chatbox             Chatbox desktop text baseline
  librechat           LibreChat text baseline
  witsy               Witsy desktop text baseline
  jan                 Jan remote OpenAI-compatible engine
  anythingllm         AnythingLLM generic OpenAI provider
  boltai              BoltAI desktop text baseline
  generic-openai      Generic OpenAI-compatible desktop/web client
  generic-mllm        Generic multimodal OpenAI-compatible client

Environment:
  VLLM_MLX_CLIENT_API_KEY  Override the default API key for the chosen profile
  LISTEN_MODE              localhost or public (passed through to serve_profile.sh)
  PORT                     Port override (passed through to serve_profile.sh)

Examples:
  scripts/serve_client_profile.sh goose-text mlx-community/Qwen3-4B-Instruct-2507-4bit
  scripts/serve_client_profile.sh goose-tools mlx-community/Qwen3-4B-Instruct-2507-4bit
  scripts/serve_client_profile.sh open-webui-mllm mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

CLIENT_PROFILE="$1"
MODEL="$2"
shift 2

if [[ ! -x "$SERVE_PROFILE_SCRIPT" ]]; then
  echo "Missing serve profile launcher: ${SERVE_PROFILE_SCRIPT}" >&2
  exit 1
fi

contains_flag() {
  local needle="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

SERVE_PROFILE=""
DEFAULT_API_KEY=""

case "$CLIENT_PROFILE" in
  goose-text)
    SERVE_PROFILE="text-default"
    DEFAULT_API_KEY="goose-local"
    ;;
  goose-tools)
    SERVE_PROFILE="text-tools"
    DEFAULT_API_KEY="goose-local"
    ;;
  open-webui-text)
    SERVE_PROFILE="text-default"
    DEFAULT_API_KEY="openwebui-local"
    ;;
  open-webui-mllm)
    SERVE_PROFILE="mllm-default"
    DEFAULT_API_KEY="openwebui-local"
    ;;
  cherry-studio)
    SERVE_PROFILE="text-default"
    DEFAULT_API_KEY="cherrystudio-local"
    ;;
  chatbox)
    SERVE_PROFILE="text-default"
    DEFAULT_API_KEY="chatbox-local"
    ;;
  librechat)
    SERVE_PROFILE="text-default"
    DEFAULT_API_KEY="librechat-local"
    ;;
  witsy)
    SERVE_PROFILE="text-default"
    DEFAULT_API_KEY="witsy-local"
    ;;
  jan)
    SERVE_PROFILE="text-default"
    DEFAULT_API_KEY="jan-local"
    ;;
  anythingllm)
    SERVE_PROFILE="text-default"
    DEFAULT_API_KEY="anythingllm-local"
    ;;
  boltai)
    SERVE_PROFILE="text-default"
    DEFAULT_API_KEY="boltai-local"
    ;;
  generic-openai)
    SERVE_PROFILE="text-default"
    DEFAULT_API_KEY="local-client"
    ;;
  generic-mllm)
    SERVE_PROFILE="mllm-default"
    DEFAULT_API_KEY="local-client"
    ;;
  *)
    echo "Unknown client profile: ${CLIENT_PROFILE}" >&2
    usage
    exit 1
    ;;
esac

API_KEY="${VLLM_MLX_CLIENT_API_KEY:-$DEFAULT_API_KEY}"

EXTRA_ARGS=("$@")
if ! contains_flag "--api-key" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
  EXTRA_ARGS+=(--api-key "$API_KEY")
fi

echo "Launching client profile '${CLIENT_PROFILE}'"
echo "Mapped serve profile: ${SERVE_PROFILE}"
echo "Default API key label: ${API_KEY}"

exec "${SERVE_PROFILE_SCRIPT}" "${SERVE_PROFILE}" "${MODEL}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
