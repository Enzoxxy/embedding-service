from typing import Any

import httpx

from app.config import Settings
from app.schemas import EmbeddingRequest


class VLLMClientError(RuntimeError):
    def __init__(self, status_code: int, message: str, response_body: Any | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class VLLMClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=settings.request_timeout_seconds)

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def embeddings(self, request: EmbeddingRequest) -> dict[str, Any]:
        payload = request.model_dump(exclude_none=True)
        return await self._request_json("POST", "/embeddings", json=payload)

    async def models(self) -> dict[str, Any]:
        return await self._request_json("GET", "/models")

    async def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.settings.vllm_base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self.settings.retry_attempts):
            try:
                response = await self.client.request(method, url, **kwargs)
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt + 1 < self.settings.retry_attempts:
                    continue
                raise VLLMClientError(503, f"vLLM backend timeout: {exc}") from exc
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt + 1 < self.settings.retry_attempts:
                    continue
                raise VLLMClientError(503, f"vLLM backend request failed: {exc}") from exc

            if response.status_code < 400:
                try:
                    return response.json()
                except ValueError as exc:
                    raise VLLMClientError(502, "vLLM returned non-JSON response") from exc

            body = _safe_json(response)
            if response.status_code < 500:
                raise VLLMClientError(response.status_code, "vLLM rejected request", body)
            last_error = VLLMClientError(response.status_code, "vLLM backend error", body)
            if attempt + 1 < self.settings.retry_attempts:
                continue
            raise last_error

        raise VLLMClientError(503, f"vLLM backend unavailable: {last_error}")


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text[:1000]

