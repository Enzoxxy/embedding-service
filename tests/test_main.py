from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, get_app_settings, get_vllm_client


class FakeVLLMClient:
    def __init__(self) -> None:
        self.last_payload = None

    async def embeddings(self, request: Any) -> dict[str, Any]:
        self.last_payload = request
        items = request.input_items()
        dim = request.dimensions or 4096
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [float(idx)] * dim, "index": idx}
                for idx, _ in enumerate(items)
            ],
            "model": request.model,
            "usage": {"prompt_tokens": len(items), "total_tokens": len(items)},
        }

    async def models(self) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "qwen3-embedding-8b"}]}


@contextmanager
def make_test_client(settings: Settings | None = None) -> Iterator[tuple[TestClient, FakeVLLMClient]]:
    fake = FakeVLLMClient()
    app.dependency_overrides[get_app_settings] = lambda: settings or Settings()
    app.dependency_overrides[get_vllm_client] = lambda: fake
    with TestClient(app) as client:
        yield client, fake
    app.dependency_overrides.clear()


def test_healthz() -> None:
    with make_test_client() as (client, _):
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models() -> None:
    with make_test_client() as (client, _):
        response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "qwen3-embedding-8b"


def test_embeddings_success_does_not_forward_default_dimensions() -> None:
    with make_test_client() as (client, fake):
        response = client.post(
            "/v1/embeddings",
            json={"model": "qwen3-embedding-8b", "input": ["a", "b"]},
        )
    assert response.status_code == 200
    body = response.json()
    assert len(body["data"]) == 2
    assert len(body["data"][0]["embedding"]) == 4096
    assert fake.last_payload.dimensions is None


def test_embeddings_forwards_explicit_dimensions() -> None:
    with make_test_client() as (client, fake):
        response = client.post(
            "/v1/embeddings",
            json={"model": "qwen3-embedding-8b", "input": "a", "dimensions": 1024},
        )
    assert response.status_code == 200
    assert len(response.json()["data"][0]["embedding"]) == 1024
    assert fake.last_payload.dimensions == 1024


def test_embeddings_rejects_wrong_model() -> None:
    with make_test_client() as (client, _):
        response = client.post("/v1/embeddings", json={"model": "other", "input": "a"})
    assert response.status_code == 400


def test_embeddings_rejects_too_many_items() -> None:
    with make_test_client(Settings(max_input_items=1)) as (client, _):
        response = client.post(
            "/v1/embeddings",
            json={"model": "qwen3-embedding-8b", "input": ["a", "b"]},
        )
    assert response.status_code == 400


def test_embeddings_query_header_rewrites_input() -> None:
    with make_test_client(Settings(query_instruction="retrieve docs")) as (client, fake):
        response = client.post(
            "/v1/embeddings",
            headers={"X-Embedding-Input-Type": "query"},
            json={"model": "qwen3-embedding-8b", "input": "怎么报销？"},
        )
    assert response.status_code == 200
    assert fake.last_payload.input == "Instruct: retrieve docs\nQuery: 怎么报销？"


def test_api_key_required_when_configured() -> None:
    with make_test_client(Settings(api_keys=["secret"])) as (client, _):
        response = client.post(
            "/v1/embeddings",
            json={"model": "qwen3-embedding-8b", "input": "a"},
        )
    assert response.status_code == 401


def test_api_key_accepts_bearer_token() -> None:
    with make_test_client(Settings(api_keys=["secret"])) as (client, _):
        response = client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer secret"},
            json={"model": "qwen3-embedding-8b", "input": "a"},
        )
    assert response.status_code == 200


def test_readyz_uses_backend_models() -> None:
    with make_test_client() as (client, _):
        response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
