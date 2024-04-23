"""Utility constants and functions used in the project."""

from dataclasses import dataclass

Index = str


@dataclass
class Document:
    """A document to be stored in a document store."""

    id: Index
    text: str
