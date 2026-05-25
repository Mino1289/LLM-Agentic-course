import re


DEFAULT_EMBEDDING_MODEL = "bge-small"
DEFAULT_COLLECTION_NAME = "semiconductor_reports"

EMBEDDING_MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
}


def resolve_embedding_model(model_key: str):
    return EMBEDDING_MODELS.get(model_key, model_key)


def collection_name_for_embedding_model(
    model_key: str,
    collection_name: str | None = None,
):
    if collection_name:
        return collection_name

    if model_key == DEFAULT_EMBEDDING_MODEL:
        return DEFAULT_COLLECTION_NAME

    safe_model_key = re.sub(r"[^a-zA-Z0-9_]+", "_", model_key).strip("_")

    return f"{DEFAULT_COLLECTION_NAME}_{safe_model_key}"
