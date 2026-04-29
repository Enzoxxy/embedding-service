#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  echo "Create it from .env.example and adjust model/runtime settings before starting." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

require_env() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "Missing required env var: ${name}" >&2
      exit 1
    fi
  done
}

require_env \
  MODEL_PATH \
  MODEL_NAME \
  CUDA_VISIBLE_DEVICES \
  VLLM_HOST \
  VLLM_PORT

exec vllm serve "${MODEL_PATH}" \
  --runner pooling \
  --convert embed \
  --served-model-name "${MODEL_NAME}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}"
