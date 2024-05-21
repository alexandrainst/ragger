"""Utility constants and functions used in the project."""

import gc
import re

import torch
from omegaconf import DictConfig

from .data_models import Components, Document


def format_answer(
    answer: str, documents: list[Document], no_documents_reply: str
) -> str:
    """Format the answer with the relevant documents.

    Args:
        answer:
            The generated answer.
        documents:
            The relevant documents.
        no_documents_reply:
            The reply to use when no documents are found.

    Returns:
        The formatted answer.
    """
    match len(documents):
        case 0:
            answer = no_documents_reply
        case 1:
            answer += "\n\nKilde:\n\n"
        case _:
            answer += "\n\nKilder:\n\n"

    formatted_ids = [
        f"<a href='{document.id}'>{document.id}</a>"
        if is_link(text=document.id)
        else document.id
        for document in documents
    ]

    answer += "\n\n".join(
        f"<details><summary>{formatted_id}</summary>{document.text}</details>"
        for formatted_id, document in zip(formatted_ids, documents)
    )
    return answer


def is_link(text: str) -> bool:
    """Check if the text is a link.

    Args:
        text:
            The text to check.

    Returns:
        Whether the text is a link.
    """
    url_regex = (
        r"^(https?:\/\/)?"  # Begins with http:// or https://, or neither
        r"(\w+\.)+"  # Then one or more blocks of lower-case letters and a dot
        r"\w{2,4}"  # Then two to four lower-case letters (e.g., .com, .dk, .org)
        r"(\/#?\w+)*?"  # Optionally followed by subdirectories or anchors
        r"(\/\w+\.\w{1,4})?"  # Optionally followed by a file suffix (e.g., .html)
    )
    return re.match(pattern=url_regex, string=text) is not None


def load_ragger_components(config: DictConfig) -> Components:
    """Load the components of the RAG system.

    Args:
        config:
            The Hydra configuration.

    """
    from .document_store import JsonlDocumentStore
    from .embedder import E5Embedder
    from .embedding_store import NumpyEmbeddingStore
    from .generator import OpenAIGenerator, VLLMGenerator

    components = Components()

    match name := config.document_store.name:
        case "jsonl":
            components.document_store = JsonlDocumentStore
        case _:
            raise ValueError(f"The DocumentStore type {name!r} is not supported")

    match name := config.embedder.name:
        case "e5":
            components.embedder = E5Embedder
        case _:
            raise ValueError(f"The Embedder type {name!r} is not supported")

    match name := config.embedding_store.name:
        case "numpy":
            components.embedding_store = NumpyEmbeddingStore
        case _:
            raise ValueError(f"The EmbeddingStore type {name!r} is not supported")

    match name := config.generator.name:
        case "openai":
            components.generator = OpenAIGenerator
        case "vllm":
            components.generator = VLLMGenerator
        case _:
            raise ValueError(f"The Generator type {name!r} is not supported")

    return components


def clear_memory() -> None:
    """Clears the memory of unused items."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
