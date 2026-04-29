#!/usr/bin/env python3
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-embedding-8b")
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "kb_documents")
VECTOR_SIZE = int(os.getenv("EMBEDDING_DIM", "4096"))
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY")


DOCUMENTS = [
    {
        "doc_id": "finance-policy",
        "chunk_id": "finance-policy-001",
        "source": "handbook",
        "title": "报销制度",
        "text": "员工报销需要提交发票、审批单和费用说明，审批通过后由财务付款。",
    },
    {
        "doc_id": "leave-policy",
        "chunk_id": "leave-policy-001",
        "source": "handbook",
        "title": "年假制度",
        "text": "员工申请年假需要提前在系统提交申请，并由直属主管审批。",
    },
]


def main() -> None:
    qdrant = QdrantClient(url=QDRANT_URL)
    ensure_collection(qdrant)

    texts = [document["text"] for document in DOCUMENTS]
    vectors = embed(texts)
    now = datetime.now(timezone.utc).isoformat()
    points = [
        qmodels.PointStruct(
            id=idx + 1,
            vector=vector,
            payload={**document, "created_at": now},
        )
        for idx, (document, vector) in enumerate(zip(DOCUMENTS, vectors))
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)

    query_vector = embed("怎么申请年假？", input_type="query")[0]
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=3,
        with_payload=True,
    )
    for result in results:
        print(
            {
                "score": result.score,
                "doc_id": result.payload.get("doc_id") if result.payload else None,
                "title": result.payload.get("title") if result.payload else None,
                "text": result.payload.get("text") if result.payload else None,
            }
        )


def ensure_collection(qdrant: QdrantClient) -> None:
    collections = qdrant.get_collections().collections
    if any(collection.name == COLLECTION for collection in collections):
        return
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE),
    )


def embed(input_value: str | list[str], input_type: str | None = None) -> list[list[float]]:
    headers = {"Content-Type": "application/json"}
    if GATEWAY_API_KEY:
        headers["Authorization"] = f"Bearer {GATEWAY_API_KEY}"
    if input_type:
        headers["X-Embedding-Input-Type"] = input_type
    with httpx.Client(timeout=120) as client:
        response = client.post(
            f"{GATEWAY_BASE_URL}/v1/embeddings",
            headers=headers,
            json={"model": MODEL_NAME, "input": input_value},
        )
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        return [item["embedding"] for item in body["data"]]


if __name__ == "__main__":
    main()
