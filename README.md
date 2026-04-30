# Embedding Service Gateway

面向本地 embedding 模型的 FastAPI 网关服务。它对外提供 OpenAI-compatible 的 `/v1/embeddings` 和 `/v1/models` 接口，后端通过 vLLM 加载本地模型并生成向量。

项目默认围绕 Qwen3-Embedding-8B 的单机单卡部署组织，但模型路径、对外模型名、向量维度、vLLM 地址、鉴权和 query instruction 都可以通过环境变量替换。

## 功能概览

- OpenAI-compatible embedding API：支持单条和批量文本输入。
- vLLM 后端代理：gateway 只负责鉴权、校验、改写和转发，向量生成由 vLLM 完成。
- Query instruction 模式：可按请求头或 `extra_body.input_type` 为检索 query 自动添加 instruction。
- API key 鉴权：支持 `Authorization: Bearer ...` 和 `X-API-Key`。
- 健康检查与就绪检查：`/healthz` 检查 gateway 存活，`/readyz` 检查 vLLM 可用性。
- Prometheus metrics：暴露请求量、延迟、输入条数和后端错误计数。
- Docker Compose 部署：可同时启动 vLLM、gateway，并可选启动 Qdrant。
- 验收脚本：部署后可自动检查 health、ready、单条 embedding、批量 embedding 和 query instruction。

## 架构

```text
client
  -> FastAPI gateway :8000
  -> VLLMClient
  -> vLLM OpenAI-compatible API :8101/v1
  -> local embedding model
```

可选 RAG 示例：

```text
client
  -> FastAPI gateway
  -> embedding vector
  -> Qdrant :6333
```

代码结构：

```text
embedding_srv/
├── app/
│   ├── main.py              # FastAPI 入口；健康检查、模型列表、embedding、metrics
│   ├── config.py            # Pydantic Settings；读取 .env 和环境变量
│   ├── schemas.py           # OpenAI-compatible 请求/响应 schema
│   ├── vllm_client.py       # 访问 vLLM /embeddings 和 /models 的异步客户端
│   └── instructions.py      # query instruction 输入改写逻辑
├── scripts/
│   ├── start.sh             # 宿主机同时启动 vLLM 和 gateway
│   ├── start_vllm.sh        # 宿主机只启动 vLLM
│   └── acceptance_smoke.py  # 部署验收脚本
├── deploy/
│   ├── docker-compose.yml   # vLLM + gateway + 可选 Qdrant
│   └── .env.example         # Docker Compose 环境变量模板
├── examples/
│   └── qdrant_demo.py       # Qdrant 写入和检索示例
├── tests/                   # 单元测试和 mock 集成测试
├── Dockerfile               # gateway 镜像
├── pyproject.toml           # Python 包、依赖和测试配置
└── README.md
```

## 环境要求

- Python `>=3.10`。
- vLLM 运行环境，需按目标机器的 GPU、CUDA、PyTorch 版本单独安装。
- 本地 embedding 模型目录，例如 `models/Qwen3-Embedding-8B`。
- Docker 部署时需要 Docker Compose、NVIDIA Container Toolkit，以及 vLLM 镜像可访问 GPU。

安装 Python 依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,examples]"
```

> `pyproject.toml` 不直接依赖 vLLM。宿主机启动 vLLM 前，需要在当前环境或系统环境中额外安装 vLLM。

## 快速启动

### 方式一：宿主机一键启动 vLLM + gateway

适合首次部署、手工验收和排障。脚本会读取项目根目录的 `.env`，先启动 vLLM，等待 `${VLLM_BASE_URL}/models` 可用后再启动 gateway。

```bash
cp .env.example .env
# 按实际模型路径、模型名、GPU 和端口修改 .env
bash scripts/start.sh
```

常见配置示例：

```dotenv
MODEL_NAME=qwen3-embedding-8b
MODEL_PATH=models/Qwen3-Embedding-8B
CUDA_VISIBLE_DEVICES=0
VLLM_HOST=0.0.0.0
VLLM_PORT=8101
VLLM_BASE_URL=http://127.0.0.1:8101/v1
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8000
EMBEDDING_DIM=4096
```

如需使用其他配置文件：

```bash
ENV_FILE=/path/to/envfile bash scripts/start.sh
```

### 方式二：只在宿主机启动 vLLM

适合 gateway 已由其他方式启动，或只想验证模型服务。当前脚本只传入模型路径、served model name、监听地址和端口。

```bash
cp .env.example .env
bash scripts/start_vllm.sh
```

### 方式三：Docker Compose 启动 vLLM + gateway

适合容器化部署。当前 compose 文件会同时启动：

- `vllm`：使用 `vllm/vllm-openai:latest` 加载本地模型。
- `gateway`：基于仓库 `Dockerfile` 构建 FastAPI gateway 镜像。
- `qdrant`：可选 profile，默认不启动。

准备 compose 环境变量：

```bash
cp deploy/.env.example deploy/.env
# 按实际模型挂载目录、模型目录名、GPU 和端口修改 deploy/.env
```

启动 vLLM 和 gateway：

```bash
docker compose --env-file deploy/.env -f deploy/docker-compose.yml up --build
```

同时启动 Qdrant：

```bash
docker compose --env-file deploy/.env -f deploy/docker-compose.yml --profile qdrant up --build
```

后台运行：

```bash
docker compose --env-file deploy/.env -f deploy/docker-compose.yml up -d --build
```

停止服务：

```bash
docker compose --env-file deploy/.env -f deploy/docker-compose.yml down
```

### 访问地址

| 组件 | 默认地址 | 说明 |
| --- | --- | --- |
| gateway | `http://127.0.0.1:8000` | 业务调用入口 |
| vLLM | `http://127.0.0.1:8101/v1` | gateway 后端，不建议直接暴露到公网 |
| Qdrant | `http://127.0.0.1:6333` | 可选向量库示例 |
| metrics | `http://127.0.0.1:8000/metrics` | Prometheus 指标 |

## API

### `GET /healthz`

gateway 存活检查，不依赖 vLLM，也不要求 API key。

```bash
curl http://127.0.0.1:8000/healthz
```

响应示例：

```json
{"status":"ok"}
```

### `GET /readyz`

gateway 就绪检查，会访问 vLLM `/models`。如果配置了 `API_KEYS`，该接口也需要鉴权。

```bash
curl http://127.0.0.1:8000/readyz
```

### `GET /v1/models`

返回 gateway 当前对外声明的模型名。如果配置了 `API_KEYS`，该接口需要鉴权。

```bash
curl http://127.0.0.1:8000/v1/models
```

响应示例：

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-embedding-8b",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

### `POST /v1/embeddings`

生成 embedding。请求中的 `model` 必须等于 gateway 配置的 `MODEL_NAME`。

单条输入：

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-8b","input":"测试文本"}'
```

批量输入：

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-8b","input":["测试文本 A","测试文本 B"]}'
```

显式指定输出维度：

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-8b","input":"测试文本","dimensions":1024}'
```

请求约束：

| 字段 | 说明 |
| --- | --- |
| `model` | 必须与 `MODEL_NAME` 一致，否则返回 `400` |
| `input` | 支持非空字符串或非空字符串数组 |
| `dimensions` | 可选，范围 `32` 到 `4096`；不传时不向 vLLM 转发该字段，由模型返回原生维度 |
| `encoding_format` | 仅支持 `float`；`base64` 会被拒绝 |
| `extra_body` | 可携带扩展字段；其中 `input_type=query` 会触发 query instruction |

单次批量输入条数受 `MAX_INPUT_ITEMS` 限制，默认 `256`。

## Query Instruction

检索场景通常需要对 query embedding 添加 instruction。gateway 支持两种方式启用。

通过请求头：

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'X-Embedding-Input-Type: query' \
  -d '{"model":"qwen3-embedding-8b","input":"怎么申请报销？"}'
```

通过 `extra_body.input_type`：

```json
{
  "model": "qwen3-embedding-8b",
  "input": "怎么申请报销？",
  "extra_body": {
    "input_type": "query"
  }
}
```

启用后，gateway 会在转发给 vLLM 前把输入改写为：

```text
Instruct: Given a user query, retrieve relevant passages that answer the query.
Query: 怎么申请报销？
```

instruction 文案由 `QUERY_INSTRUCTION` 控制。

## 鉴权

`API_KEYS` 为空时不启用鉴权。配置一个或多个 key 后，`/readyz`、`/v1/models` 和 `/v1/embeddings` 需要携带有效 key。`/healthz` 和 `/metrics` 不做鉴权。

`.env` 示例：

```dotenv
API_KEYS=key-a,key-b
```

Bearer token：

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Authorization: Bearer key-a' \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-8b","input":"测试文本"}'
```

`X-API-Key`：

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'X-API-Key: key-a' \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-8b","input":"测试文本"}'
```

## 配置

### 应用配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_NAME` | `qwen3-embedding-8b` | gateway 对外声明的模型名；请求 `model` 必须与它一致 |
| `MODEL_PATH` | `models/Qwen3-Embedding-8B` | 本地模型路径；也用于启动脚本 |
| `VLLM_BASE_URL` | `http://127.0.0.1:8101/v1` | gateway 访问 vLLM 的 OpenAI-compatible base URL |
| `EMBEDDING_DIM` | `4096` | 预期向量维度，主要用于验收脚本和 Qdrant 示例 |
| `MAX_INPUT_ITEMS` | `256` | 单次 embedding 请求最多输入条数 |
| `REQUEST_TIMEOUT_SECONDS` | `120` | gateway 访问 vLLM 的请求超时时间 |
| `RETRY_ATTEMPTS` | `2` | gateway 访问 vLLM 的最大尝试次数 |
| `API_KEYS` | 空 | 逗号分隔的 API key 列表，空表示不启用鉴权 |
| `QUERY_INSTRUCTION` | `Given a user query, retrieve relevant passages that answer the query.` | query instruction 文案 |

### 宿主机启动脚本配置

`scripts/start.sh` 要求 `.env` 中存在以下变量：

| 变量 | 说明 |
| --- | --- |
| `MODEL_PATH` | vLLM 加载的模型路径 |
| `MODEL_NAME` | vLLM served model name，也是 gateway 对外模型名 |
| `CUDA_VISIBLE_DEVICES` | 使用的 GPU ID |
| `VLLM_HOST` | vLLM 监听地址 |
| `VLLM_PORT` | vLLM 监听端口 |
| `VLLM_BASE_URL` | gateway 访问 vLLM 的地址 |
| `VLLM_DTYPE` | `scripts/start.sh` 传给 vLLM 的 dtype |
| `VLLM_MAX_MODEL_LEN` | `scripts/start.sh` 传给 vLLM 的最大上下文长度 |
| `VLLM_GPU_MEMORY_UTILIZATION` | `scripts/start.sh` 传给 vLLM 的显存利用率 |
| `VLLM_STARTUP_TIMEOUT_SECONDS` | 等待 vLLM 就绪的超时时间 |
| `GATEWAY_HOST` | gateway 监听地址 |
| `GATEWAY_PORT` | gateway 监听端口 |
| `GATEWAY_APP` | uvicorn 启动的 ASGI 应用，例如 `app.main:app` |

### Docker Compose 配置

`deploy/docker-compose.yml` 建议配合 `deploy/.env` 使用：

| 变量 | 说明 |
| --- | --- |
| `MODEL_PATH` | 宿主机上的模型父目录，会挂载到容器 `/models` |
| `MODEL_DIR_NAME` | `MODEL_PATH` 下的模型目录名 |
| `MODEL_NAME` | vLLM served model name 和 gateway 对外模型名 |
| `CUDA_VISIBLE_DEVICES` | vLLM 容器可见 GPU |
| `VLLM_PORT` | 映射到宿主机的 vLLM 端口 |
| `GATEWAY_PORT` | 映射到宿主机的 gateway 端口 |
| `QDRANT_PORT` | 可选 Qdrant HTTP 端口 |

如果模型目录是 `/data/models/Qwen3-Embedding-8B`，compose 配置应类似：

```dotenv
MODEL_PATH=/data/models
MODEL_DIR_NAME=Qwen3-Embedding-8B
MODEL_NAME=qwen3-embedding-8b
```

## Metrics

`GET /metrics` 暴露 Prometheus 文本格式指标，当前不做鉴权。

主要指标：

| 指标 | 说明 |
| --- | --- |
| `embedding_gateway_requests_total` | 按 endpoint 和 status 统计请求数 |
| `embedding_gateway_request_seconds` | 按 endpoint 统计请求耗时 |
| `embedding_gateway_input_items_total` | embedding 输入条目总数 |
| `embedding_gateway_backend_errors_total` | 返回给客户端的后端错误数 |

## Qdrant 示例

启动 gateway 和 Qdrant 后运行：

```bash
python examples/qdrant_demo.py
```

带 gateway 鉴权：

```bash
GATEWAY_API_KEY=key-a python examples/qdrant_demo.py
```

示例会：

1. 连接 `QDRANT_URL`，默认 `http://127.0.0.1:6333`。
2. 创建 collection，默认名称 `kb_documents`，向量维度来自 `EMBEDDING_DIM`。
3. 写入两条示例文档。
4. 使用 query instruction 生成检索 query 向量。
5. 在 Qdrant 中执行 cosine 相似度检索并打印结果。

可配置变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `GATEWAY_BASE_URL` | `http://127.0.0.1:8000` | gateway 地址 |
| `MODEL_NAME` | `qwen3-embedding-8b` | 请求模型名 |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant 地址 |
| `QDRANT_COLLECTION` | `kb_documents` | collection 名称 |
| `EMBEDDING_DIM` | `4096` | collection 向量维度 |
| `GATEWAY_API_KEY` | 空 | gateway API key |

## 部署验收

服务启动后运行：

```bash
python scripts/acceptance_smoke.py
```

指定地址、模型名和维度：

```bash
python scripts/acceptance_smoke.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3-embedding-8b \
  --expected-dim 4096
```

带鉴权：

```bash
GATEWAY_API_KEY=key-a python scripts/acceptance_smoke.py
```

同时运行 Qdrant 示例：

```bash
python scripts/acceptance_smoke.py --qdrant
```

验收内容：

- `/healthz`
- `/readyz`
- 单条 embedding
- 批量 embedding
- query instruction embedding
- 可选 Qdrant 示例

结果会写入：

```text
reports/acceptance/YYYYMMDD-HHMMSS/
├── summary.json
├── requests.jsonl
├── vectors_meta.json
└── env.txt
```

## 测试

单元测试和 mock 集成测试不需要真实 vLLM、模型或 GPU。

```bash
pytest
```

测试覆盖范围包括：

- schema 校验：空输入、维度范围、`encoding_format` 限制。
- API 行为：模型名校验、批量限制、鉴权、query instruction、readyz。
- vLLM client：请求路径、4xx 不重试、5xx 重试、超时映射为 `503`。

## 错误处理与重试

- vLLM `4xx` 错误会直接透传为对应客户端错误，不重试。
- vLLM `5xx` 错误会按 `RETRY_ATTEMPTS` 重试，最终对外返回后端错误；未知 `5xx` 会映射为 `502`。
- vLLM 请求超时或网络错误会映射为 `503`。
- 请求 schema 错误、模型名不匹配、输入为空或超出批量限制会返回 `400`。

## 生产建议

- 对公网提供服务时，在 gateway 前增加 nginx、API gateway 或负载均衡，用于 TLS、限流、访问日志和更细粒度访问控制。
- vLLM 后端建议只暴露在内网或容器网络中，由 gateway 统一对外提供 API。
- 生产镜像建议固定 vLLM 镜像 tag，不使用 `latest`。
- 根据模型实际输出维度同步设置 `EMBEDDING_DIM`，否则验收脚本和 Qdrant collection 维度会不匹配。
- 监控 `/metrics`，重点关注 `embedding_gateway_backend_errors_total`、延迟分布和输入条数。
- 宿主机脚本适合排障和单机部署；长期运行建议交给 systemd、supervisor 或容器编排系统托管。
