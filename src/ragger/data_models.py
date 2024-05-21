"""Data models used in the RAG system."""

from typing import TYPE_CHECKING, Annotated, Type

import annotated_types
import numpy as np
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .document_store import DocumentStore
    from .embedder import Embedder
    from .embedding_store import EmbeddingStore
    from .generator import Generator


Index = str


class Document(BaseModel):
    """A document to be stored in a document store."""

    id: Index
    text: str


class Embedding(BaseModel):
    """An embedding of a document."""

    id: Index
    embedding: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GeneratedAnswer(BaseModel):
    """A generated answer to a question."""

    sources: list[Annotated[Index, annotated_types.Len(min_length=1)]]
    answer: str = ""


class Components(BaseModel):
    """The components of the RAG system."""

    document_store: "Type[DocumentStore] | None" = None
    embedder: "Type[Embedder] | None" = None
    embedding_store: "Type[EmbeddingStore] | None" = None
    generator: "Type[Generator] | None" = None
