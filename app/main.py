from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.instructions import apply_instruction_if_needed
from app.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    ModelData,
    ModelsResponse,
)
from app.vllm_client import VLLMClient, VLLMClientError


REQUEST_COUNT = Counter(
    "embedding_gateway_requests_total",
    "Total embedding gateway requests.",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "embedding_gateway_request_seconds",
    "Embedding gateway request latency.",
    ["endpoint"],
)
INPUT_ITEMS = Counter(
    "embedding_gateway_input_items_total",
    "Total embedding input items.",
)
BACKEND_ERRORS = Counter(
    "embedding_gateway_backend_errors_total",
    "Total backend errors returned to clients.",
    ["status"],
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.vllm_client = VLLMClient(settings)
    try:
        yield
    finally:
        await app.state.vllm_client.close()


app = FastAPI(title="Qwen3 Embedding Gateway", version="0.1.0", lifespan=lifespan)


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_vllm_client(request: Request) -> VLLMClient:
    return request.app.state.vllm_client


async def require_api_key(
    settings: Annotated[Settings, Depends(get_app_settings)],
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    if not settings.api_keys:
        return
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    token = token or x_api_key
    if token not in settings.api_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    REQUEST_COUNT.labels("/healthz", "200").inc()
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(
    _: Annotated[None, Depends(require_api_key)],
    client: Annotated[VLLMClient, Depends(get_vllm_client)],
) -> dict[str, Any]:
    start = perf_counter()
    try:
        models = await client.models()
    except VLLMClientError as exc:
        REQUEST_COUNT.labels("/readyz", str(exc.status_code)).inc()
        BACKEND_ERRORS.labels(str(exc.status_code)).inc()
        raise HTTPException(status_code=503, detail={"status": "not_ready", "error": str(exc)}) from exc
    finally:
        REQUEST_LATENCY.labels("/readyz").observe(perf_counter() - start)
    REQUEST_COUNT.labels("/readyz", "200").inc()
    return {"status": "ready", "backend_models": models.get("data", [])}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models", response_model=ModelsResponse)
async def models(
    _: Annotated[None, Depends(require_api_key)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> ModelsResponse:
    REQUEST_COUNT.labels("/v1/models", "200").inc()
    return ModelsResponse(data=[ModelData(id=settings.model_name)])


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(
    raw_request: Request,
    _: Annotated[None, Depends(require_api_key)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    client: Annotated[VLLMClient, Depends(get_vllm_client)],
    x_embedding_input_type: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    start = perf_counter()
    try:
        payload = await raw_request.json()
        embedding_request = EmbeddingRequest.model_validate(payload)
        if embedding_request.model != settings.model_name:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported model '{embedding_request.model}', expected '{settings.model_name}'",
            )
        if len(embedding_request.input_items()) > settings.max_input_items:
            raise HTTPException(
                status_code=400,
                detail=f"input item count exceeds max_input_items={settings.max_input_items}",
            )
        if embedding_request.dimensions is None:
            embedding_request = embedding_request.model_copy(update={"dimensions": settings.embedding_dim})
        embedding_request = apply_instruction_if_needed(
            embedding_request,
            settings,
            x_embedding_input_type,
        )
        INPUT_ITEMS.inc(len(embedding_request.input_items()))
        response = await client.embeddings(embedding_request)
    except ValidationError as exc:
        REQUEST_COUNT.labels("/v1/embeddings", "400").inc()
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except VLLMClientError as exc:
        status_code = _public_status_code(exc.status_code)
        REQUEST_COUNT.labels("/v1/embeddings", str(status_code)).inc()
        BACKEND_ERRORS.labels(str(status_code)).inc()
        raise HTTPException(
            status_code=status_code,
            detail={"message": str(exc), "backend_response": exc.response_body},
        ) from exc
    except HTTPException as exc:
        REQUEST_COUNT.labels("/v1/embeddings", str(exc.status_code)).inc()
        raise
    finally:
        REQUEST_LATENCY.labels("/v1/embeddings").observe(perf_counter() - start)

    REQUEST_COUNT.labels("/v1/embeddings", "200").inc()
    response["model"] = settings.model_name
    return response


def _public_status_code(status_code: int) -> int:
    if status_code in {400, 401, 403, 404, 422, 429, 503}:
        return status_code
    if 500 <= status_code:
        return 502
    return status_code

