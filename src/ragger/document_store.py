"""Store and fetch documents from a database."""

import json
import sqlite3
import typing
from contextlib import contextmanager
from pathlib import Path

from .data_models import Document, DocumentStore, Index


class JsonlDocumentStore(DocumentStore):
    """A document store that fetches documents from a JSONL file."""

    def __init__(self, path: Path = Path("document-store.jsonl")) -> None:
        """Initialise the document store.

        Args:
            path (optional):
                The path to the JSONL file where the documents are stored. Defaults to
                "document-store.jsonl".
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

    def add_documents(self, documents: typing.Iterable[Document]) -> typing.Self:
        """Add documents to the store.

        Args:
            documents:
                An iterable of documents to add to the store.
        """
        for document in documents:
            self._documents[document.id] = document

        # Write the documents to the file
        data_str = "\n".join(document.model_dump_json() for document in documents)
        self.path.write_text(data_str)

        return self

    def remove(self) -> None:
        """Remove the document store."""
        self.path.unlink(missing_ok=True)

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

    def __iter__(self) -> typing.Generator[Document, None, None]:
        """Iterate over the documents in the store.

        Yields:
            The documents in the store.
        """
        yield from self._documents.values()

    def __len__(self) -> int:
        """Return the number of documents in the store.

        Returns:
            The number of documents in the store.
        """
        return len(self._documents)


class SqliteDocumentStore(DocumentStore):
    """A document store that fetches documents from a SQLite database."""

    def __init__(self, path: Path = Path("document-store.sqlite")) -> None:
        """Initialise the document store.

        Args:
            path:
                The path to the SQLite database where the documents are stored.
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    text TEXT
                )
                """
            )

    @contextmanager
    def _connect(self) -> typing.Generator[sqlite3.Connection, None, None]:
        """Connect to the SQLite database.

        Yields:
            The connection to the database.
        """
        conn = sqlite3.connect(self.path)
        try:
            yield conn
        finally:
            conn.close()

    def add_documents(self, documents: typing.Iterable[Document]) -> typing.Self:
        """Add documents to the store.

        Args:
            documents:
                An iterable of documents to add to the store.
        """
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO documents (id, text) VALUES (?, ?)",
                [(document.id, document.text) for document in documents],
            )
            conn.commit()
        return self

    def remove(self) -> None:
        """Remove the document store."""
        self.path.unlink(missing_ok=True)

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
        with self._connect() as conn:
            cursor = conn.execute("SELECT text FROM documents WHERE id = ?", (index,))
            row = cursor.fetchone()
            if row is None:
                raise KeyError(f"Document with ID {index!r} not found")
            return Document(id=index, text=row[0])

    def __contains__(self, index: Index) -> bool:
        """Check if a document with the given ID exists in the store.

        Args:
            index:
                The ID of the document to check.

        Returns:
            Whether the document exists in the store.
        """
        with self._connect() as conn:
            cursor = conn.execute("SELECT 1 FROM documents WHERE id = ?", (index,))
            return cursor.fetchone() is not None

    def __iter__(self) -> typing.Generator[Document, None, None]:
        """Iterate over the documents in the store.

        Yields:
            The documents in the store.
        """
        with self._connect() as conn:
            cursor = conn.execute("SELECT id, text FROM documents")
            for row in cursor:
                yield Document(id=row[0], text=row[1])

    def __len__(self) -> int:
        """Return the number of documents in the store.

        Returns:
            The number of documents in the store.
        """
        with self._connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]
