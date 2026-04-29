import httpx
import pytest

from app.config import Settings
from app.schemas import EmbeddingRequest
from app.vllm_client import VLLMClient, VLLMClientError


@pytest.mark.asyncio
async def test_embeddings_success() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
                "model": "qwen3-embedding-8b",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    transport = httpx.MockTransport(handler)
    client = VLLMClient(
        Settings(vllm_base_url="http://test/v1"),
        httpx.AsyncClient(transport=transport),
    )
    response = await client.embeddings(EmbeddingRequest(model="qwen3-embedding-8b", input="text"))
    assert response["data"][0]["embedding"] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_4xx_is_not_retried() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = VLLMClient(
        Settings(vllm_base_url="http://test/v1", retry_attempts=2),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(VLLMClientError) as exc:
        await client.embeddings(EmbeddingRequest(model="qwen3-embedding-8b", input="text"))
    assert exc.value.status_code == 400
    assert calls == 1


@pytest.mark.asyncio
async def test_5xx_is_retried() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, json={"error": "temporary"})
        return httpx.Response(200, json={"data": [], "model": "qwen3-embedding-8b"})

    client = VLLMClient(
        Settings(vllm_base_url="http://test/v1", retry_attempts=2),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    response = await client.embeddings(EmbeddingRequest(model="qwen3-embedding-8b", input="text"))
    assert response["model"] == "qwen3-embedding-8b"
    assert calls == 2


@pytest.mark.asyncio
async def test_timeout_maps_to_503() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout")

    client = VLLMClient(
        Settings(vllm_base_url="http://test/v1", retry_attempts=1),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(VLLMClientError) as exc:
        await client.embeddings(EmbeddingRequest(model="qwen3-embedding-8b", input="text"))
    assert exc.value.status_code == 503

