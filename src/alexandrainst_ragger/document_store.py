"""Store and fetch documents from a database."""

import json
import sqlite3
import typing
from contextlib import contextmanager
from pathlib import Path

from .data_models import Document, DocumentStore, Index
from .utils import is_installed, raise_if_not_installed

if is_installed(package_name="psycopg2"):
    import psycopg2

if typing.TYPE_CHECKING:
    import psycopg2


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

    def add_documents(self, documents: typing.Iterable[Document]) -> "DocumentStore":
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
        self._documents.clear()
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

    def __init__(
        self,
        path: Path = Path("document-store.sqlite"),
        table_name: str = "documents",
        id_column: str = "id",
        text_column: str = "text",
    ) -> None:
        """Initialise the document store.

        Args:
            path:
                The path to the SQLite database where the documents are stored.
            table_name (optional):
                The name of the table in the database where the documents are stored.
                Defaults to "documents".
            id_column (optional):
                The name of the column in the table that stores the document IDs.
                Defaults to "id".
            text_column (optional):
                The name of the column in the table that stores the document text.
                Defaults to "text".
        """
        self.path = path
        self.table_name = table_name
        self.id_column = id_column
        self.text_column = text_column

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {id_column} TEXT PRIMARY KEY,
                    {text_column} TEXT
                )
                """
            )
            conn.commit()

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

    def add_documents(self, documents: typing.Iterable[Document]) -> "DocumentStore":
        """Add documents to the store.

        Args:
            documents:
                An iterable of documents to add to the store.
        """
        with self._connect() as conn:
            conn.executemany(
                f"""
                INSERT OR REPLACE INTO {self.table_name} (
                    {self.id_column},
                    {self.text_column}
                ) VALUES (?, ?)
                """,
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
            cursor = conn.execute(
                f"""
                SELECT text FROM {self.table_name}
                WHERE {self.id_column} = ?
            """,
                (index,),
            )
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
            cursor = conn.execute(
                f"""
                SELECT 1 FROM {self.table_name}
                WHERE {self.id_column} = ?
            """,
                (index,),
            )
            return cursor.fetchone() is not None

    def __iter__(self) -> typing.Generator[Document, None, None]:
        """Iterate over the documents in the store.

        Yields:
            The documents in the store.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                f"SELECT {self.id_column}, {self.text_column} FROM {self.table_name}"
            )
            for row in cursor:
                yield Document(id=row[0], text=row[1])

    def __len__(self) -> int:
        """Return the number of documents in the store.

        Returns:
            The number of documents in the store.
        """
        with self._connect() as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            return cursor.fetchone()[0]


class PostgresDocumentStore(DocumentStore):
    """A document store that fetches documents from a PostgreSQL database."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str | None = "postgres",
        password: str | None = "postgres",
        database_name: str = "postgres",
        table_name: str = "documents",
        id_column: str = "id",
        text_column: str = "text",
    ) -> None:
        """Initialise the document store.

        Args:
            host (optional):
                The hostname of the PostgreSQL database. Defaults to "localhost".
            port (optional):
                The port of the PostgreSQL database. Defaults to 5432.
            user (optional):
                The username to connect to the PostgreSQL database. Defaults to
                "postgres".
            password (optional):
                The password to connect to the PostgreSQL database. Defaults to
                "postgres".
            database_name (optional):
                The name of the database where the documents are stored. Defaults to
                "postgres".
            table_name (optional):
                The name of the table in the database where the documents are stored.
                Defaults to "documents".
            id_column (optional):
                The name of the column in the table that stores the document IDs.
                Defaults to "id".
            text_column (optional):
                The name of the column in the table that stores the document text.
                Defaults to "text".
        """
        raise_if_not_installed(
            package_names=["psycopg2"],
            extra="postgres",
            installation_alias_mapping=dict(psycopg2="psycopg2-binary"),
        )

        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database_name = database_name
        self.table_name = table_name
        self.id_column = id_column
        self.text_column = text_column

        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"CREATE DATABASE {database_name}")
            except psycopg2.errors.DuplicateDatabase:
                pass
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {id_column} TEXT PRIMARY KEY,
                    {text_column} TEXT
                )
                """
            )

    @contextmanager
    def _connect(
        self,
    ) -> "typing.Generator[psycopg2.extensions.connection, None, None]":
        """Connect to the PostgreSQL database.

        Yields:
            The connection to the database.
        """
        connection = psycopg2.connect(
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            dbname=self.database_name,
        )
        connection.autocommit = True
        yield connection
        connection.close()

    def add_documents(self, documents: typing.Iterable[Document]) -> "DocumentStore":
        """Add documents to the store.

        Args:
            documents:
                An iterable of documents to add to the store.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                f"""
                INSERT INTO {self.table_name} (
                    {self.id_column},
                    {self.text_column}
                ) VALUES (%s, %s)
                ON CONFLICT ({self.id_column}) DO UPDATE SET
                    {self.text_column} = EXCLUDED.{self.text_column}
                """,
                [(document.id, document.text) for document in documents],
            )
        return self

    def remove(self) -> None:
        """Remove the document store."""
        with self._connect() as conn:
            conn.cursor().execute(f"DROP TABLE IF EXISTS {self.table_name}")

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
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {self.text_column} FROM {self.table_name}
                WHERE {self.id_column} = %s
            """,
                (index,),
            )
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
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT 1 FROM {self.table_name}
                WHERE {self.id_column} = %s
            """,
                (index,),
            )
            return cursor.fetchone() is not None

    def __iter__(self) -> typing.Generator[Document, None, None]:
        """Iterate over the documents in the store.

        Yields:
            The documents in the store.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {self.id_column}, {self.text_column} FROM {self.table_name}"
            )
            for row in cursor:
                yield Document(id=row[0], text=row[1])

    def __len__(self) -> int:
        """Return the number of documents in the store.

        Returns:
            The number of documents in the store.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            except psycopg2.errors.UndefinedTable:
                return 0
            else:
                result = cursor.fetchone()
                if result is None:
                    return 0
                return result[0]


class TxtDocumentStore(DocumentStore):
    """A document store that fetches documents from a TXT file."""

    def __init__(self, path: Path = Path("document-store.txt")) -> None:
        """Initialise the document store.

        Args:
            path (optional):
                The path to the TXT file where the documents are stored. Defaults to
                "document-store.txt".
        """
        self.path = path

        # Ensure the file exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

        lines = [line for line in self.path.read_text().splitlines() if line.strip()]
        self._documents = {
            str(i): Document(id=str(i), text=line) for i, line in enumerate(lines)
        }

    def add_documents(self, documents: typing.Iterable[Document]) -> "DocumentStore":
        """Add documents to the store.

        Args:
            documents:
                An iterable of documents to add to the store.
        """
        for document in documents:
            self._documents[document.id] = document

        # Write the documents to the file
        with self.path.open("a") as file:
            for document in documents:
                file.write(document.text + "\n")

        return self

    def remove(self) -> None:
        """Remove the document store."""
        self._documents.clear()
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
