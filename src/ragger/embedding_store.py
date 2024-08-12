"""Store and fetch embeddings from a database."""

import io
import json
import logging
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data_models import Embedding, EmbeddingStore, Index

logger = logging.getLogger(__package__)


class NumpyEmbeddingStore(EmbeddingStore):
    """An embedding store that fetches embeddings from a NumPy file."""

    def __init__(
        self, embedding_dim: int, path: Path = Path("embedding-store.zip")
    ) -> None:
        """Initialise the NumPy embedding store.

        Args:
            embedding_dim:
                The dimension of the embeddings.
            path:
                The path to the zipfile where the embeddings are stored.
        """
        self.path = path
        self.embedding_dim = embedding_dim
        self.embeddings = np.zeros((0, self.embedding_dim))
        self.index_to_row_id: dict[Index, int] = defaultdict()
        self.load_embeddings_if_exists()

    def load_embeddings_if_exists(self) -> None:
        """Load the embeddings from disk if they exist."""
        if self.path and self.path.exists():
            self.load(path=self.path)

    @property
    def row_id_to_index(self) -> dict[int, Index]:
        """Return a mapping of row IDs to indices."""
        return {row_id: index for index, row_id in self.index_to_row_id.items()}

    def add_embeddings(self, embeddings: list[Embedding]) -> None:
        """Add embeddings to the store.

        Args:
            embeddings:
                A list of embeddings to add to the store.

        Raises:
            ValueError:
                If any of the embeddings already exist in the store.
        """
        if not embeddings:
            return

        logger.info(f"Adding {len(embeddings):,} embeddings to the embedding store...")

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
            return

        embedding_matrix = np.stack(
            arrays=[embedding.embedding for embedding in embeddings]
        )
        self.embeddings = np.vstack([self.embeddings, embedding_matrix])

        for i, embedding in enumerate(embeddings):
            self.index_to_row_id[embedding.id] = (
                self.embeddings.shape[0] - len(embeddings) + i
            )

        logger.info("Added embeddings to the embedding store.")

        if self.path is not None:
            self.save(path=self.path)

    def save(self, path: Path | str) -> None:
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

    def load(self, path: Path | str) -> None:
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
        # Ensure that the number of documents to retrieve is less than the number of
        # documents in the store
        num_docs = max(num_docs, self.embeddings.shape[0])

        logger.info(f"Finding {num_docs:,} nearest neighbours...")
        scores = self.embeddings @ embedding
        top_indices = np.argsort(scores)[::-1][:num_docs]
        nearest_neighbours = [self.row_id_to_index[i] for i in top_indices]
        logger.info(f"Found nearest neighbours with indices {top_indices}.")
        return nearest_neighbours

    def document_exists_in_store(self, document_id: Index) -> bool:
        """Check if a document exists in the store.

        Args:
            document_id:
                The ID of the document to check.

        Returns:
            Whether the document exists in the store.
        """
        return document_id in self.index_to_row_id

    def clear(self) -> None:
        """Clear all embeddings from the store."""
        self.embeddings = np.zeros(shape=(0, self.embedding_dim))
        self.index_to_row_id = defaultdict()
        if self.path:
            self.path.unlink(missing_ok=True)
        logger.info("Cleared the embedding store.")
