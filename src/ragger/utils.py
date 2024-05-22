"""Utility constants and functions used in the project."""

import gc
import importlib
import os
import re
import sys
from typing import Type

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


def clear_memory() -> None:
    """Clears the memory of unused items."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def snake_to_pascal(snake_string: str) -> str:
    """Converts a snake_case string to PascalCase.

    Args:
        snake_string:
            The snake_case string.

    Returns:
        The PascalCase string.
    """
    return "".join(word.title() for word in snake_string.split("_"))


def get_component_by_name(class_name: str, component_type: str) -> Type:
    """Get a component class by its name.

    Args:
        class_name:
            The name of the class, written in snake_case.
        component_type:
            The type of component, written in snake_case. It is assumed that a module
            exists with this name in the `ragger` package.

    Returns:
        The class.

    Raises:
        ValueError:
            If the module or class cannot be found.
    """
    # Get the snake_case and PascalCase version of the class name
    full_class_name = f"{class_name}_{component_type}"
    name_pascal = snake_to_pascal(snake_string=full_class_name)

    # Get the module
    module_name = f"ragger.{component_type}"
    try:
        module = importlib.import_module(name=module_name)
    except ModuleNotFoundError:
        raise ValueError(f"Module {module_name!r}' not found.")

    # Get the class from the module
    try:
        class_: Type = getattr(module, name_pascal)
    except AttributeError:
        raise ValueError(f"Class {name_pascal!r} not found in module {module_name!r}.")

    return class_


def load_ragger_components(config: DictConfig) -> Components:
    """Load the components of the RAG system.

    Args:
        config:
            The Hydra configuration.

    """
    return Components(
        document_store=get_component_by_name(
            class_name=config.document_store.name, component_type="document_store"
        ),
        embedder=get_component_by_name(
            class_name=config.embedder.name, component_type="embedder"
        ),
        embedding_store=get_component_by_name(
            class_name=config.embedding_store.name, component_type="embedding_store"
        ),
        generator=get_component_by_name(
            class_name=config.generator.name, component_type="generator"
        ),
    )


class HiddenPrints:
    """Context manager which removes all terminal output."""

    def __enter__(self):
        """Enter the context manager."""
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
