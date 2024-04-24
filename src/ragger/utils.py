"""Utility constants and functions used in the project."""

import numpy as np
from pydantic import BaseModel, ConfigDict

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
    sources: list[Index]
