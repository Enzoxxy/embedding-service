#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

MODEL_PATH="${MODEL_PATH:-models/Qwen3-Embedding-8B}"
MODEL_NAME="${MODEL_NAME:-qwen3-embedding-8b}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8101}"
VLLM_CONNECT_HOST="${VLLM_CONNECT_HOST:-127.0.0.1}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_CONNECT_HOST}:${VLLM_PORT}/v1}"
VLLM_DTYPE="${VLLM_DTYPE:-float16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-600}"

GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"
GATEWAY_APP="${GATEWAY_APP:-app.main:app}"

VLLM_PID=""
GATEWAY_PID=""

cleanup() {
  local exit_code=$?
  trap - INT TERM EXIT
  if [[ -n "${GATEWAY_PID}" ]] && kill -0 "${GATEWAY_PID}" 2>/dev/null; then
    kill "${GATEWAY_PID}" 2>/dev/null || true
  fi
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
  wait "${GATEWAY_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
  exit "${exit_code}"
}

trap cleanup INT TERM EXIT

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MODEL_NAME
export MODEL_PATH
export VLLM_BASE_URL

echo "Starting vLLM on ${VLLM_HOST}:${VLLM_PORT} with GPU ${CUDA_VISIBLE_DEVICES}"
vllm serve "${MODEL_PATH}" \
  --runner pooling \
  --convert embed \
  --served-model-name "${MODEL_NAME}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --dtype "${VLLM_DTYPE}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" &
VLLM_PID=$!

echo "Waiting for vLLM readiness at ${VLLM_BASE_URL}/models"
deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SECONDS))
until curl -fsS "${VLLM_BASE_URL}/models" >/dev/null 2>&1; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready" >&2
    wait "${VLLM_PID}"
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for vLLM after ${VLLM_STARTUP_TIMEOUT_SECONDS}s" >&2
    exit 1
  fi
  sleep 2
done

echo "Starting gateway on ${GATEWAY_HOST}:${GATEWAY_PORT}"
uvicorn "${GATEWAY_APP}" --host "${GATEWAY_HOST}" --port "${GATEWAY_PORT}" &
GATEWAY_PID=$!

echo "Gateway is starting. Public embedding endpoint: http://${GATEWAY_HOST}:${GATEWAY_PORT}/v1/embeddings"
wait -n "${VLLM_PID}" "${GATEWAY_PID}"
