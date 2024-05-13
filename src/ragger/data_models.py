"""Data models used in the RAG system."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

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

    answer: str
    sources: list[Index] = Field(default_factory=list)
