import pytest
from pydantic import ValidationError

from app.schemas import EmbeddingRequest


def test_accepts_string_input() -> None:
    request = EmbeddingRequest(model="qwen3-embedding-8b", input="测试文本")
    assert request.input_items() == ["测试文本"]


def test_accepts_list_input() -> None:
    request = EmbeddingRequest(model="qwen3-embedding-8b", input=["a", "b"])
    assert request.input_items() == ["a", "b"]


@pytest.mark.parametrize("input_value", ["", "   ", [], ["ok", ""]])
def test_rejects_empty_input(input_value: object) -> None:
    with pytest.raises(ValidationError):
        EmbeddingRequest(model="qwen3-embedding-8b", input=input_value)


@pytest.mark.parametrize("dimensions", [31, 4097])
def test_rejects_invalid_dimensions(dimensions: int) -> None:
    with pytest.raises(ValidationError):
        EmbeddingRequest(model="qwen3-embedding-8b", input="text", dimensions=dimensions)


def test_rejects_base64_encoding() -> None:
    with pytest.raises(ValidationError):
        EmbeddingRequest(model="qwen3-embedding-8b", input="text", encoding_format="base64")

