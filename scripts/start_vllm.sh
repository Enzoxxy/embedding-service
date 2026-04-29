#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/Qwen3-Embedding-8B}"
MODEL_NAME="${MODEL_NAME:-qwen3-embedding-8b}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8101}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

exec vllm serve "${MODEL_PATH}" \
  --runner pooling \
  --convert embed \
  --served-model-name "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}"
