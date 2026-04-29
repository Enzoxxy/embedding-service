# Qwen3-Embedding-8B Embedding Service

FastAPI gateway for a single local `Qwen3-Embedding-8B` vLLM backend. The public API is OpenAI-compatible at `/v1/embeddings`.

## Defaults

- Model path: `models/Qwen3-Embedding-8B`
- Public model name: `qwen3-embedding-8b`
- vLLM backend: `http://127.0.0.1:8101/v1`
- Gateway: `http://0.0.0.0:8000`
- Embedding dimension: `4096`
- GPU plan: one RTX3090 only, no multi-backend routing, no tensor parallel

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,examples]"
```

Install vLLM separately on the target GPU server according to its CUDA/PyTorch environment.

## Start Service

Start vLLM and the gateway together:

```bash
bash scripts/start.sh
```

The combined script starts vLLM first, waits for `http://127.0.0.1:8101/v1/models`, then starts the gateway on port `8000`.

`scripts/start.sh` automatically loads `.env` from the project root when the file exists. Already exported shell variables override script defaults through normal shell behavior.

Useful overrides:

```bash
MODEL_PATH=../models/Qwen3-Embedding-8B \
VLLM_MAX_MODEL_LEN=8192 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
bash scripts/start.sh
```

## Start vLLM Only

```bash
bash scripts/start_vllm.sh
```

The script uses:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve models/Qwen3-Embedding-8B \
  --runner pooling \
  --convert embed \
  --served-model-name qwen3-embedding-8b \
  --host 0.0.0.0 \
  --port 8101
```

## Start Gateway

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Request

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-8b","input":["测试文本"]}'
```

Query instruction mode is opt-in:

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'X-Embedding-Input-Type: query' \
  -d '{"model":"qwen3-embedding-8b","input":"怎么申请报销？"}'
```

If `API_KEYS` is configured on the gateway, pass `Authorization: Bearer <key>` or `X-API-Key: <key>`.

## Acceptance On Server

This local project does not execute real GPU/vLLM/Qdrant acceptance. After deploying on the server, run:

```bash
python scripts/acceptance_smoke.py
```

With gateway authentication:

```bash
GATEWAY_API_KEY=your-key python scripts/acceptance_smoke.py
```

It writes records under `reports/acceptance/YYYYMMDD-HHMMSS/`:

- `summary.json`
- `requests.jsonl`
- `vectors_meta.json`
- `env.txt`

These files are intended for feedback and debugging if deployment fails.

## Qdrant Example

Start Qdrant, then run:

```bash
python examples/qdrant_demo.py
```

With gateway authentication:

```bash
GATEWAY_API_KEY=your-key python examples/qdrant_demo.py
```

The example creates a `4096`-dimensional cosine collection named `kb_documents`, writes two sample chunks, and runs one search query.

## Tests

Unit tests and mock integration tests do not require vLLM or GPU:

```bash
pytest
```
