"""Store and fetch embeddings from a database."""

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

from .data_models import Embedding, EmbeddingStore, Index
from .utils import is_installed, raise_if_not_installed

if is_installed(package_name="psycopg2"):
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
        if (
            self.embedding_dim is None
            or self.embeddings is None
            or self.embeddings.size == 0
        ):
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
            return list()

        # Ensure that the number of documents to retrieve is less than the number of
        # documents in the store
        num_docs = min(num_docs, len(self))

        logger.info(f"Finding {num_docs:,} nearest neighbours...")
        scores = self.embeddings @ embedding
        top_indices = np.argsort(scores)[::-1][:num_docs]
        nearest_neighbours = [self.row_id_to_index[i] for i in top_indices]
        logger.info(f"Found nearest neighbours with indices {top_indices}.")
        return nearest_neighbours

    def remove(self) -> None:
        """Remove the embedding store."""
        if self.embedding_dim is not None:
            self.embeddings = np.zeros(shape=(0, self.embedding_dim))
        self.index_to_row_id = defaultdict()
        self.path.unlink(missing_ok=True)
        logger.info("The embedding store has been removed.")

    def __getitem__(self, document_id: Index) -> Embedding:
        """Fetch an embedding by its document ID.

        Args:
            document_id:
                The ID of the document to fetch the embedding for.

        Returns:
            The embedding with the given document ID.

        Raises:
            KeyError:
                If the document ID does not exist in the store, or if the store is empty.
        """
        if self.embeddings is None:
            raise KeyError("The store is empty.")
        if document_id not in self.index_to_row_id:
            raise KeyError(f"The document ID {document_id!r} does not exist.")
        row_id = self.index_to_row_id[document_id]
        embedding = self.embeddings[row_id]
        return Embedding(id=document_id, embedding=embedding)

    def __contains__(self, document_id: Index) -> bool:
        """Check if a document exists in the store.

        Args:
            document_id:
                The ID of the document to check.

        Returns:
            Whether the document exists in the store.
        """
        return document_id in self.index_to_row_id

    def __iter__(self) -> typing.Generator[Embedding, None, None]:
        """Iterate over the embeddings in the store.

        Yields:
            The embeddings in the store.
        """
        if self.embeddings is None:
            return
        for document_id, row_id in self.index_to_row_id.items():
            embedding = self.embeddings[row_id]
            yield Embedding(id=document_id, embedding=embedding)

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
        embedding_dim: int | None = None,
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
            embedding_dim (optional):
                The dimension of the embeddings. If None then the dimension will be
                inferred when embeddings are added. Defaults to None.
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
        raise_if_not_installed(
            package_names=["psycopg2"],
            extra="postgres",
            installation_alias_mapping=dict(psycopg2="psycopg2-binary"),
        )

        self.embedding_dim = embedding_dim
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

        self._create_table()

    def _create_table(self) -> None:
        """Create the table in the database."""
        if self.embedding_dim is None:
            return
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    {self.id_column} TEXT PRIMARY KEY,
                    {self.embedding_column} VECTOR({self.embedding_dim})
                )
            """)
            cursor.execute(f"""
                ALTER TABLE {self.table_name}
                ADD COLUMN IF NOT EXISTS {self.embedding_column} VECTOR({self.embedding_dim})
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS cosine_hnsw_embedding_idx
                ON {self.table_name}
                USING hnsw ({self.embedding_column} vector_cosine_ops)
            """)

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

    def add_embeddings(
        self, embeddings: typing.Iterable[Embedding]
    ) -> "EmbeddingStore":
        """Add embeddings to the store.

        Args:
            embeddings:
                An iterable of embeddings to add to the store.
        """
        if not embeddings:
            return self

        # Ensure that the table exists
        self._create_table()

        # Ensure that we can access the embeddings multiple times
        embeddings = list(embeddings)

        if self.embedding_dim is None:
            self.embedding_dim = embeddings[0].embedding.shape[0]
            self._create_table()

        id_embedding_pairs = [
            (embedding.id, json.dumps(embedding.embedding.tolist()))
            for embedding in embeddings
        ]

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                f"""
                INSERT INTO {self.table_name} ({self.id_column}, {self.embedding_column})
                VALUES (%s, %s)
                ON CONFLICT ({self.id_column}) DO UPDATE
                SET {self.embedding_column} = EXCLUDED.{self.embedding_column}
                """,
                id_embedding_pairs,
            )

        return self

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
        """
        if self.embedding_dim is None:
            return list()

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {self.id_column}
                FROM {self.table_name}
                ORDER BY {self.embedding_column} <=> %s
                LIMIT {num_docs}
                """,
                (json.dumps(embedding.tolist()),),
            )
            return [row[0] for row in cursor.fetchall()]

    def __getitem__(self, document_id: Index) -> Embedding:
        """Fetch an embedding by its document ID.

        Args:
            document_id:
                The ID of the document to fetch the embedding for.

        Returns:
            The embedding with the given document ID.

        Raises:
            KeyError:
                If the document ID does not exist in the store, or if the store is empty.
        """
        if self.embedding_dim is None:
            raise KeyError("The store is empty.")

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {self.embedding_column}
                FROM {self.table_name}
                WHERE {self.id_column} = %s
                """,
                (document_id,),
            )
            result = cursor.fetchone()
            if result is None:
                raise KeyError(f"The document ID {document_id!r} does not exist.")
            embedding = np.asarray(json.loads(result[0]))
            return Embedding(id=document_id, embedding=embedding)

    def __contains__(self, document_id: Index) -> bool:
        """Check if a document exists in the store.

        Args:
            document_id:
                The ID of the document to check.

        Returns:
            Whether the document exists in the store.
        """
        if self.embedding_dim is None:
            return False
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

    def __iter__(self) -> typing.Generator[Embedding, None, None]:
        """Iterate over the embeddings in the store.

        Yields:
            The embeddings in the store.
        """
        if self.embedding_dim is None:
            return
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {self.id_column}, {self.embedding_column}
                FROM {self.table_name}
                WHERE {self.embedding_column} IS NOT NULL
                """
            )
            for row in cursor.fetchall():
                embedding = np.asarray(json.loads(row[1]))
                yield Embedding(id=row[0], embedding=embedding)

    def __len__(self) -> int:
        """Return the number of embeddings in the store.

        Returns:
            The number of embeddings in the store.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"""
                    SELECT COUNT(*)
                    FROM {self.table_name}
                    WHERE {self.embedding_column} IS NOT NULL
                """)
            except (psycopg2.errors.UndefinedTable, psycopg2.errors.UndefinedColumn):
                return 0
            else:
                result = cursor.fetchone()
                if result is None:
                    return 0
                return result[0]

    def remove(self) -> None:
        """Clear all embeddings from the store."""
        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"""
                    ALTER TABLE {self.table_name} DROP COLUMN {self.embedding_column}
                """)
            except (psycopg2.errors.UndefinedTable, psycopg2.errors.UndefinedColumn):
                pass
