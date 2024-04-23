"""Utility constants and functions used in the project."""

from pydantic import BaseModel

Index = str


class Document(BaseModel):
    """A document to be stored in a document store."""

    id: Index
    text: str
