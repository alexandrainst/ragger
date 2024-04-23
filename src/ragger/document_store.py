"""Store and fetch documents from a database."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from omegaconf import DictConfig

from .utils import Document, Index


class DocumentStore(ABC):
    """An abstract document store, which fetches documents from a database."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the document store.

        Args:
            config:
                The Hydra configuration.
        """
        self.config = config

    @abstractmethod
    def __getitem__(self, index: Index) -> Document:
        """Fetch a document by its ID.

        Args:
            index:
                The ID of the document to fetch.

        Returns:
            The document with the given ID.
        """
        ...


class JsonlDocumentStore(DocumentStore):
    """A document store that fetches documents from a JSONL file."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the document store.

        Args:
            config:
                The Hydra configuration.
        """
        super().__init__(config)
        data_path = Path(self.config.document_store_filename)
        data_dicts = [
            json.loads(line)
            for line in data_path.read_text().splitlines()
            if line.strip()
        ]
        self._documents = {
            dct["id"]: Document.model_validate(dct) for dct in data_dicts
        }

    def __getitem__(self, index: Index) -> Document:
        """Fetch a document by its ID.

        Args:
            index:
                The ID of the document to fetch.

        Returns:
            The document with the given ID.

        Raises:
            KeyError:
                If the document with the given ID is not found.
        """
        if index not in self._documents:
            raise KeyError(f"Document with ID {index!r} not found")
        return self._documents[index]