"""Store and fetch documents from a database."""

import json
from pathlib import Path

from .data_models import Document, DocumentStore, Index


class JsonlDocumentStore(DocumentStore):
    """A document store that fetches documents from a JSONL file."""

    def __init__(self, path: Path = Path("document-store.jsonl")) -> None:
        """Initialise the document store.

        Args:
            path:
                The path to the JSONL file where the documents are stored.
        """
        self.path = path

        # Ensure the file exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

        data_dicts = [
            json.loads(line)
            for line in self.path.read_text().splitlines()
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

    def __contains__(self, index: Index) -> bool:
        """Check if a document with the given ID exists in the store.

        Args:
            index:
                The ID of the document to check.

        Returns:
            Whether the document exists in the store.
        """
        return index in self._documents

    def get_all_documents(self) -> list[Document]:
        """Fetch all documents from the store.

        Returns:
            A list of all documents in the store.
        """
        return list(self._documents.values())
