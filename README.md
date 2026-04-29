# Embedding Service Gateway

这是一个可替换 embedding 模型的基础服务框架。项目用 FastAPI 提供 OpenAI-compatible 的 `/v1/embeddings` API，后端通过 vLLM 加载本地 embedding 模型。

当前仓库内置了一组可直接运行的默认示例配置；实际部署时可以通过环境变量替换模型路径、对外模型名、向量维度和 query instruction。

默认示例配置：

- 模型路径：`models/Qwen3-Embedding-8B`
- 对外模型名：`qwen3-embedding-8b`
- vLLM 后端：`http://127.0.0.1:8101/v1`
- FastAPI 网关：`http://0.0.0.0:8000`
- 向量维度：`4096`
- 部署形态：单 vLLM 后端、单 gateway；脚本默认按单卡 vLLM 启动

## 代码架构

```text
embedding_srv/
├── app/
│   ├── main.py              # FastAPI 应用入口，定义健康检查、模型列表、embedding、metrics API
│   ├── config.py            # 环境变量和默认配置
│   ├── schemas.py           # 请求/响应 Pydantic schema
│   ├── vllm_client.py       # 访问 vLLM OpenAI-compatible API 的异步客户端
│   └── instructions.py      # query instruction 输入改写逻辑
├── scripts/
│   ├── start.sh             # 同时启动 vLLM 和 FastAPI 网关
│   ├── start_vllm.sh        # 只启动 vLLM 后端
│   └── acceptance_smoke.py  # 服务器验收脚本
├── deploy/
│   └── docker-compose.yml   # 容器化启动 gateway，可选启动 Qdrant
├── examples/
│   └── qdrant_demo.py       # Qdrant 写入和检索示例
├── tests/                   # 单元测试和 mock 集成测试
├── pyproject.toml           # Python 包、依赖和测试配置
└── README.md
```

核心调用链：

```text
client
  -> FastAPI gateway :8000
  -> VLLMClient
  -> vLLM OpenAI API :8101/v1
  -> embedding model
```

可选 RAG 示例调用链：

```text
client
  -> FastAPI gateway
  -> embedding vector
  -> Qdrant :6333
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,examples]"
```

vLLM 需要根据目标 GPU 服务器的 CUDA/PyTorch 环境单独安装。

## 启动命令

### 方案一：宿主机一键启动 vLLM + gateway

适合首次部署、验收和排障。脚本会先启动 vLLM，等待 `http://127.0.0.1:8101/v1/models` 可用后，再启动 FastAPI 网关。

```bash
bash scripts/start.sh
```

常用覆盖参数：

```bash
MODEL_PATH=/path/to/embedding-model \
MODEL_NAME=my-embedding-model \
EMBEDDING_DIM=1024 \
VLLM_MAX_MODEL_LEN=8192 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
bash scripts/start.sh
```

`scripts/start.sh` 会自动读取项目根目录下的 `.env`。已经在 shell 中导出的环境变量优先级更高。

### 方案二：宿主机启动 vLLM + Docker Compose 启动 gateway

适合希望将 FastAPI 网关容器化、并可选拉起 Qdrant 的部署方式。vLLM 仍然运行在宿主机上，gateway 容器通过 `host.docker.internal:8101` 访问宿主机 vLLM。

先启动 vLLM：

```bash
bash scripts/start_vllm.sh
```

再启动 gateway 容器：

```bash
docker compose -f deploy/docker-compose.yml up gateway
```

如果还需要 Qdrant：

```bash
docker compose -f deploy/docker-compose.yml --profile qdrant up
```

当前 `deploy/docker-compose.yml` 中的 gateway 容器会在启动时执行 `pip install -e .`。生产环境建议改成固定镜像，避免每次启动都重新安装依赖。

## 服务与 API 连接说明矩阵

| 服务/组件 | 默认地址 | 启动方式 | 调用方 | 主要用途 | 备注 |
| --- | --- | --- | --- | --- | --- |
| FastAPI gateway | `http://0.0.0.0:8000` | `scripts/start.sh` 或 `docker compose ... up gateway` | 外部客户端、验收脚本、Qdrant 示例 | 对外提供 embedding API、模型列表、健康检查和 metrics | 实际业务调用入口 |
| vLLM backend | `http://127.0.0.1:8101/v1` | `scripts/start.sh` 或 `scripts/start_vllm.sh` | FastAPI gateway | 加载本地 embedding 模型并生成向量 | 不建议直接暴露给外网 |
| Qdrant | `http://127.0.0.1:6333` | `docker compose --profile qdrant up` | `examples/qdrant_demo.py` 或业务系统 | 向量存储和相似度检索 | 可选组件，不是 gateway 必需依赖 |
| Prometheus metrics | `http://<gateway>:8000/metrics` | 随 gateway 启动 | Prometheus 或监控系统 | 暴露请求量、延迟、输入条数和后端错误指标 | 无鉴权逻辑 |
| 验收脚本 | 连接 `GATEWAY_BASE_URL`，默认 `http://127.0.0.1:8000` | `python scripts/acceptance_smoke.py` | 运维/部署人员 | 检查健康、就绪、单条 embedding、批量 embedding、query instruction | 输出到 `reports/acceptance/` |

## 对外 API

### `GET /healthz`

网关存活检查，不依赖 vLLM 后端。

```bash
curl http://127.0.0.1:8000/healthz
```

### `GET /readyz`

网关就绪检查，会访问 vLLM `/models`。如果 vLLM 不可用，会返回非 200。

```bash
curl http://127.0.0.1:8000/readyz
```

### `GET /v1/models`

返回当前 gateway 对外声明的模型名。

```bash
curl http://127.0.0.1:8000/v1/models
```

### `POST /v1/embeddings`

OpenAI-compatible embedding API。

下面示例使用默认示例模型名 `qwen3-embedding-8b`。如果替换了模型，请同步替换请求中的 `model` 字段，确保它和 `MODEL_NAME` 一致。

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-8b","input":["测试文本"]}'
```

支持单条输入：

```json
{
  "model": "qwen3-embedding-8b",
  "input": "测试文本"
}
```

支持批量输入：

```json
{
  "model": "qwen3-embedding-8b",
  "input": ["测试文本 A", "测试文本 B"]
}
```

支持显式指定输出维度，范围是 `32` 到 `4096`：

```json
{
  "model": "qwen3-embedding-8b",
  "input": "测试文本",
  "dimensions": 1024
}
```

当前 gateway 只支持 `encoding_format=float`，不支持 `base64`。

### Query instruction 模式

检索场景下，query embedding 通常需要附加 instruction。可以通过请求头启用：

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'X-Embedding-Input-Type: query' \
  -d '{"model":"qwen3-embedding-8b","input":"怎么申请报销？"}'
```

也可以通过 `extra_body.input_type` 启用：

```json
{
  "model": "qwen3-embedding-8b",
  "input": "怎么申请报销？",
  "extra_body": {
    "input_type": "query"
  }
}
```

启用后，gateway 会把输入改写为：

```text
Instruct: Given a user query, retrieve relevant passages that answer the query.
Query: 怎么申请报销？
```

## 鉴权

如果配置了 `API_KEYS`，gateway 会要求请求携带 API key。支持两种方式：

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Authorization: Bearer your-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-8b","input":"测试文本"}'
```

或：

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'X-API-Key: your-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-8b","input":"测试文本"}'
```

如果 `API_KEYS` 为空，则不启用鉴权。

## 关键环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_NAME` | `qwen3-embedding-8b` | gateway 对外声明的模型名，同时传给 vLLM served model name；替换模型时需要同步调整 |
| `MODEL_PATH` | `models/Qwen3-Embedding-8B` | 本地模型路径；替换模型时指向新的模型目录 |
| `VLLM_BASE_URL` | `http://127.0.0.1:8101/v1` | gateway 访问 vLLM 的 OpenAI-compatible API 地址 |
| `EMBEDDING_DIM` | `4096` | 预期向量维度，主要用于验收和示例；替换模型时应改成新模型的实际输出维度 |
| `MAX_INPUT_ITEMS` | `256` | 单次 embedding 请求最多输入条数 |
| `REQUEST_TIMEOUT_SECONDS` | `120` | gateway 访问 vLLM 的请求超时时间 |
| `RETRY_ATTEMPTS` | `2` | gateway 访问 vLLM 的重试次数 |
| `API_KEYS` | 空 | 逗号分隔的 API key 列表 |
| `QUERY_INSTRUCTION` | `Given a user query, retrieve relevant passages that answer the query.` | query instruction 模式使用的提示词；可按模型建议或业务场景调整 |

## Qdrant 示例

启动 Qdrant 后运行：

```bash
python examples/qdrant_demo.py
```

带 gateway 鉴权时：

```bash
GATEWAY_API_KEY=your-key python examples/qdrant_demo.py
```

示例会创建一个与 `EMBEDDING_DIM` 一致的 cosine collection：`kb_documents`，写入两条样例文档，并用 query instruction 模式执行一次检索。默认示例配置下维度为 `4096`。

## 服务器验收

部署完成后运行：

```bash
python scripts/acceptance_smoke.py
```

带 gateway 鉴权时：

```bash
GATEWAY_API_KEY=your-key python scripts/acceptance_smoke.py
```

如果需要同时跑 Qdrant 示例：

```bash
python scripts/acceptance_smoke.py --qdrant
```

验收结果会写入：

```text
reports/acceptance/YYYYMMDD-HHMMSS/
├── summary.json
├── requests.jsonl
├── vectors_meta.json
└── env.txt
```

## 测试

单元测试和 mock 集成测试不需要真实 vLLM 或 GPU：

```bash
pytest
```

## 生产部署建议

- 首次部署优先使用 `scripts/start.sh` 跑通模型、GPU、vLLM 和 gateway。
- 长期运行时建议用 systemd、supervisor 或容器编排系统分别守护 vLLM 和 gateway。
- 如果使用 Docker Compose 方案，建议为 gateway 构建固定镜像，不要在容器启动时临时 `pip install -e .`。
- vLLM 后端建议只监听内网或本机地址，由 gateway 统一对外提供 API。
- 对公网服务时建议在 gateway 前增加 nginx/API gateway，用于 TLS、限流、访问控制和日志采集。
