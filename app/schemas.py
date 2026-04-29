from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EmbeddingInput = str | list[str]


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: EmbeddingInput
    encoding_format: Literal["float", "base64"] | None = "float"
    dimensions: int | None = Field(default=None, ge=32, le=4096)
    user: str | None = None
    extra_body: dict[str, Any] | None = None

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: EmbeddingInput) -> EmbeddingInput:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("input must not be empty")
            return value
        if not value:
            raise ValueError("input list must not be empty")
        empty_indexes = [idx for idx, item in enumerate(value) if not item.strip()]
        if empty_indexes:
            raise ValueError(f"input items must not be empty: {empty_indexes}")
        return value

    @model_validator(mode="after")
    def validate_encoding(self) -> "EmbeddingRequest":
        if self.encoding_format not in (None, "float"):
            raise ValueError("only encoding_format=float is supported by this gateway")
        return self

    def input_items(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else self.input


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage = Field(default_factory=Usage)


class ModelData(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "local"


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelData]


class ErrorResponse(BaseModel):
    error: dict[str, Any]

