"""Store and fetch embeddings from a database."""

import importlib.util
import io
import json
import logging
import typing
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np

from .data_models import (
    DocumentStore,
    Embedder,
    Embedding,
    EmbeddingStore,
    Generator,
    Index,
)

if importlib.util.find_spec("psycopg2") is not None:
    import psycopg2

if typing.TYPE_CHECKING:
    import psycopg2

logger = logging.getLogger(__package__)


class NumpyEmbeddingStore(EmbeddingStore):
    """An embedding store that fetches embeddings from a NumPy file."""

    def __init__(
        self, embedding_dim: int | None = None, path: Path = Path("embedding-store.zip")
    ) -> None:
        """Initialise the NumPy embedding store.

        Args:
            embedding_dim (optional):
                The dimension of the embeddings. If None then the dimension will be
                inferred when embeddings are added. Defaults to None.
            path (optional):
                The path to the zipfile where the embeddings are stored. Defaults to
                "embedding-store.zip".
        """
        self.path = path
        self.embedding_dim = embedding_dim
        self.embeddings: np.ndarray | None = None
        self.index_to_row_id: dict[Index, int] = defaultdict()
        self._initialise_embedding_matrix()
        if self.path.exists():
            self._load(path=self.path)

    def _initialise_embedding_matrix(self) -> None:
        """Initialise the embedding matrix with zeros."""
        if self.embedding_dim is not None:
            self.embeddings = np.zeros((0, self.embedding_dim))

    def compile(
        self,
        document_store: "DocumentStore",
        embedder: "Embedder",
        generator: "Generator",
    ) -> None:
        """Compile the embedding store by adding all embeddings from the document store.

        Args:
            document_store:
                The document store to use.
            embedder:
                The embedder to use.
            generator:
                The generator to use.
        """
        documents_not_in_embedding_store = [
            document for document in document_store if document.id not in self
        ]
        embeddings = embedder.embed_documents(
            documents=documents_not_in_embedding_store
        )
        self.add_embeddings(embeddings=embeddings)

    @property
    def row_id_to_index(self) -> dict[int, Index]:
        """Return a mapping of row IDs to indices."""
        return {row_id: index for index, row_id in self.index_to_row_id.items()}

    def add_embeddings(self, embeddings: Iterable[Embedding]) -> "EmbeddingStore":
        """Add embeddings to the store.

        Args:
            embeddings:
                An iterable of embeddings to add to the store.

        Raises:
            ValueError:
                If any of the embeddings already exist in the store.
        """
        if not embeddings:
            return self

        already_existing_indices = [
            embedding.id
            for embedding in embeddings
            if embedding.id in self.index_to_row_id
        ]
        if already_existing_indices:
            num_already_existing_indices = len(already_existing_indices)
            logger.warning(
                f"{num_already_existing_indices:,} embeddings already existed in the "
                "embedding store and was ignored."
            )

        embeddings = [
            embedding
            for embedding in embeddings
            if embedding.id not in self.index_to_row_id
        ]
        if not embeddings:
            return self

        # In case we haven't inferred the embedding dimension yet, we do it now
        if self.embedding_dim is None or self.embeddings is None:
            self.embedding_dim = embeddings[0].embedding.shape[0]
            self._initialise_embedding_matrix()
        assert self.embeddings is not None

        logger.info(f"Adding {len(embeddings):,} embeddings to the embedding store...")

        embedding_matrix = np.stack(
            arrays=[embedding.embedding for embedding in embeddings]
        )
        self.embeddings = np.vstack([self.embeddings, embedding_matrix])

        for i, embedding in enumerate(embeddings):
            self.index_to_row_id[embedding.id] = (
                self.embeddings.shape[0] - len(embeddings) + i
            )

        logger.info("Added embeddings to the embedding store.")

        self._save(path=self.path)
        return self

    def _save(self, path: Path | str) -> None:
        """Save the embedding store to disk.

        Args:
            path:
                The path to the embeddings store in. This should be a .zip-file. This
                zip file will contain the embeddings matrix in the file `embeddings.npy`
                and the row ID to index mapping in the file `index_to_row_id.json`.

        Raises:
            ValueError:
                If the path is not a zip file.
        """
        if self.embeddings is None:
            return

        logger.info(f"Saving embeddings to {path!r}...")

        path = Path(path)
        if path.suffix != ".zip":
            raise ValueError("The path must be a zip file.")

        array_file = io.BytesIO()
        np.save(file=array_file, arr=self.embeddings)

        index_to_row_id = json.dumps(self.index_to_row_id).encode("utf-8")

        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("embeddings.npy", data=array_file.getvalue())
            zf.writestr("index_to_row_id.json", data=index_to_row_id)

        logger.info("Saved embeddings.")

    def _load(self, path: Path | str) -> None:
        """This loads the embeddings store from disk.

        Args:
            path:
                The path to the zip file to load the embedding store from.
        """
        logger.info(f"Loading embeddings from {str(path)!r}...")
        path = Path(path)
        with zipfile.ZipFile(file=path, mode="r") as zf:
            index_to_row_id_encoded = zf.read("index_to_row_id.json")
            index_to_row_id = json.loads(index_to_row_id_encoded.decode("utf-8"))
            array_file = io.BytesIO(zf.read("embeddings.npy"))
            embeddings = np.load(file=array_file, allow_pickle=False)
        self.embeddings = embeddings
        self.embedding_dim = embeddings.shape[1]
        self.index_to_row_id = index_to_row_id
        self.row_id_to_index
        logger.info("Loaded embeddings.")

    def get_nearest_neighbours(
        self, embedding: np.ndarray, num_docs: int = 5
    ) -> list[Index]:
        """Get the nearest neighbours to a given embedding.

        Args:
            embedding:
                The embedding to find nearest neighbours for.
            num_docs (optional):
                The number of documents to retrieve. Defaults to 5.

        Returns:
            A list of indices of the nearest neighbours.

        Raises:
            ValueError:
                If the number of documents in the store is less than the number of
                documents to retrieve.
        """
        if self.embeddings is None:
            return []

        # Ensure that the number of documents to retrieve is less than the number of
        # documents in the store
        num_docs = max(num_docs, self.embeddings.shape[0])

        logger.info(f"Finding {num_docs:,} nearest neighbours...")
        scores = self.embeddings @ embedding
        top_indices = np.argsort(scores)[::-1][:num_docs]
        nearest_neighbours = [self.row_id_to_index[i] for i in top_indices]
        logger.info(f"Found nearest neighbours with indices {top_indices}.")
        return nearest_neighbours

    def clear(self) -> None:
        """Clear all embeddings from the store."""
        if self.embedding_dim is not None:
            self.embeddings = np.zeros(shape=(0, self.embedding_dim))
        self.index_to_row_id = defaultdict()
        self.path.unlink(missing_ok=True)
        logger.info("Cleared the embedding store.")

    def remove(self) -> None:
        """Remove the embedding store."""
        self.path.unlink(missing_ok=True)

    def __contains__(self, document_id: Index) -> bool:
        """Check if a document exists in the store.

        Args:
            document_id:
                The ID of the document to check.

        Returns:
            Whether the document exists in the store.
        """
        return document_id in self.index_to_row_id

    def __len__(self) -> int:
        """Return the number of embeddings in the store.

        Returns:
            The number of embeddings in the store.
        """
        if self.embeddings is None:
            return 0
        return self.embeddings.shape[0]


class PostgresEmbeddingStore(EmbeddingStore):
    """An embedding store that fetches embeddings from a PostgreSQL database."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str | None = "postgres",
        password: str | None = "postgres",
        database_name: str = "postgres",
        table_name: str = "embeddings",
        id_column: str = "id",
        embedding_column: str = "embedding",
    ) -> None:
        """Initialise the PostgreSQL embedding store.

        Args:
            host (optional):
                The hostname of the PostgreSQL server. Defaults to "localhost".
            port (optional):
                The port of the PostgreSQL server. Defaults to 5432.
            user (optional):
                The username to use when connecting to the PostgreSQL server. Defaults
                to "postgres".
            password (optional):
                The password to use when connecting to the PostgreSQL server. Defaults
                to "postgres".
            database_name (optional):
                The name of the database to use. Defaults to "postgres".
            table_name (optional):
                The name of the table to use. Defaults to "documents".
            id_column (optional):
                The name of the column containing the document IDs. Defaults to "id".
            embedding_column (optional):
                The name of the column containing the embeddings. Defaults to
                "embedding".
        """
        psycopg2_not_installed = importlib.util.find_spec("psycopg2") is None
        if psycopg2_not_installed:
            raise ImportError(
                "The `postgres` extra is required to use the `PostgresDocumentStore`. "
                "Please install it by running `pip install ragger[postgres]@"
                "git+ssh://git@github.com/alexandrainst/ragger.git` and try again."
            )

        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database_name = database_name
        self.table_name = table_name
        self.id_column = id_column
        self.embedding_column = embedding_column

        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"CREATE DATABASE {database_name}")
            except psycopg2.errors.DuplicateDatabase:
                pass
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except psycopg2.errors.UniqueViolation:
                pass
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {id_column} TEXT PRIMARY KEY,
                    {embedding_column} VECTOR
                )
                """
            )

    @contextmanager
    def _connect(self) -> typing.Generator[psycopg2.extensions.connection, None, None]:
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

    def add_embeddings(
        self, embeddings: typing.Iterable[Embedding]
    ) -> "EmbeddingStore":
        """Add embeddings to the store.

        Args:
            embeddings:
                An iterable of embeddings to add to the store.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                f"""
                INSERT INTO {self.table_name} ({self.id_column}, {self.embedding_column})
                VALUES (%s, %s)
                """,
                [(embedding.id, embedding.embedding) for embedding in embeddings],
            )
        return self

    def get_nearest_neighbours(self, embedding: np.ndarray) -> list[Index]:
        """Get the nearest neighbours to a given embedding.

        Args:
            embedding:
                The embedding to find nearest neighbours for.

        Returns:
            A list of indices of the nearest neighbours.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {self.id_column}
                FROM {self.table_name}
                ORDER BY {self.embedding_column} <=> %s
                LIMIT 5
                """,
                (embedding,),
            )
            return [row[0] for row in cursor.fetchall()]

    def __contains__(self, document_id: Index) -> bool:
        """Check if a document exists in the store.

        Args:
            document_id:
                The ID of the document to check.

        Returns:
            Whether the document exists in the store.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT 1
                FROM {self.table_name}
                WHERE {self.id_column} = %s
                """,
                (document_id,),
            )
            return cursor.fetchone() is not None

    def __len__(self) -> int:
        """Return the number of embeddings in the store.

        Returns:
            The number of embeddings in the store.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            result = cursor.fetchone()
            assert result is not None
            return result[0]

    def clear(self) -> None:
        """Clear all embeddings from the store."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE {self.table_name} SET {self.embedding_column} = NULL"
            )

    def remove(self) -> None:
        """Remove the embedding store."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE {self.table_name}")
