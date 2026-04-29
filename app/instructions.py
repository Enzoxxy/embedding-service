from app.config import Settings
from app.schemas import EmbeddingRequest


QUERY_INPUT_TYPE = "query"


def resolve_input_type(request: EmbeddingRequest, header_input_type: str | None) -> str | None:
    if header_input_type:
        return header_input_type.strip().lower()
    if request.extra_body:
        input_type = request.extra_body.get("input_type")
        if isinstance(input_type, str):
            return input_type.strip().lower()
    return None


def apply_instruction_if_needed(
    request: EmbeddingRequest,
    settings: Settings,
    header_input_type: str | None = None,
) -> EmbeddingRequest:
    input_type = resolve_input_type(request, header_input_type)
    if input_type != QUERY_INPUT_TYPE:
        return request

    def format_query(text: str) -> str:
        return f"Instruct: {settings.query_instruction}\nQuery: {text}"

    updated_input = (
        format_query(request.input)
        if isinstance(request.input, str)
        else [format_query(item) for item in request.input]
    )
    return request.model_copy(update={"input": updated_input})

