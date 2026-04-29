from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_name: str = "qwen3-embedding-8b"
    model_path: str = "models/Qwen3-Embedding-8B"
    vllm_base_url: str = "http://127.0.0.1:8101/v1"
    embedding_dim: Annotated[int, Field(ge=32, le=4096)] = 4096
    max_input_items: Annotated[int, Field(ge=1)] = 256
    request_timeout_seconds: Annotated[float, Field(gt=0)] = 120
    retry_attempts: Annotated[int, Field(ge=1, le=5)] = 2
    api_keys: list[str] = Field(default_factory=list)
    query_instruction: str = "Given a user query, retrieve relevant passages that answer the query."

    @field_validator("vllm_base_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, value: object) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise TypeError("api_keys must be a comma separated string or list")


@lru_cache
def get_settings() -> Settings:
    return Settings()

