from app.config import Settings
from app.instructions import apply_instruction_if_needed
from app.schemas import EmbeddingRequest


def test_default_does_not_rewrite_input() -> None:
    settings = Settings()
    request = EmbeddingRequest(model=settings.model_name, input="怎么报销？")
    updated = apply_instruction_if_needed(request, settings)
    assert updated.input == "怎么报销？"


def test_header_query_input_type_adds_instruction() -> None:
    settings = Settings(query_instruction="retrieve relevant docs")
    request = EmbeddingRequest(model=settings.model_name, input="怎么报销？")
    updated = apply_instruction_if_needed(request, settings, "query")
    assert updated.input == "Instruct: retrieve relevant docs\nQuery: 怎么报销？"


def test_extra_body_query_input_type_adds_instruction_for_batch() -> None:
    settings = Settings(query_instruction="retrieve relevant docs")
    request = EmbeddingRequest(
        model=settings.model_name,
        input=["怎么报销？", "怎么请假？"],
        extra_body={"input_type": "query"},
    )
    updated = apply_instruction_if_needed(request, settings)
    assert updated.input == [
        "Instruct: retrieve relevant docs\nQuery: 怎么报销？",
        "Instruct: retrieve relevant docs\nQuery: 怎么请假？",
    ]

