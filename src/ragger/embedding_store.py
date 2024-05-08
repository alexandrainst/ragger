"""Store and fetch embeddings from a database."""

import io
import json
import logging
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from transformers import AutoConfig

from .data_models import Embedding, Index

logger = logging.getLogger(__package__)


class EmbeddingStore(ABC):
    """An abstract embedding store, which fetches embeddings from a database."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the embedding store.

        Args:
            config:
                The Hydra configuration.
        """
        self.config = config

    @abstractmethod
    def add_embeddings(self, embeddings: list[Embedding]) -> None:
        """Add embeddings to the store.

        Args:
            embeddings:
                A list of embeddings to add to the store.
        """
        ...

    @abstractmethod
    def get_nearest_neighbours(self, embedding: np.ndarray) -> list[Index]:
        """Get the nearest neighbours to a given embedding.

        Args:
            embedding:
                The embedding to find nearest neighbours for.

        Returns:
            A list of indices of the nearest neighbours.
        """
        ...


class NumpyEmbeddingStore(EmbeddingStore):
    """An embedding store that fetches embeddings from a NumPy file."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the NumPy embedding store.

        Args:
            config:
                The Hydra configuration.
        """
        super().__init__(config)
        self.embedding_dim = self._get_embedding_dimension()
        self.embeddings = np.zeros((0, self.embedding_dim))
        self.index_to_row_id: dict[Index, int] = defaultdict()
        self.load_embeddings_if_exists()

    def load_embeddings_if_exists(self) -> None:
        """Load the embeddings from disk if they exist."""
        if self.config.embedding_store.embedding_path is None:
            return
        embedding_store_path = (
            Path(self.config.dirs.data) / self.config.embedding_store.embedding_path
        )
        if embedding_store_path.exists():
            self.load(path=embedding_store_path)

    @property
    def row_id_to_index(self) -> dict[int, Index]:
        """Return a mapping of row IDs to indices."""
        return {row_id: index for index, row_id in self.index_to_row_id.items()}

    def _get_embedding_dimension(self) -> int:
        """This returns the embedding dimension for the embedding model.

        Returns:
            The embedding dimension.
        """
        model_config = AutoConfig.from_pretrained(
            self.config.embedder.model_id, cache_dir=self.config.dirs.models
        )
        return model_config.hidden_size

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
        embedding_matrix = np.stack(
            [
                embedding.embedding
                for embedding in embeddings
                if embedding.id not in self.index_to_row_id
            ]
        )

        self.embeddings = np.vstack([self.embeddings, embedding_matrix])
        for i, embedding in enumerate(embeddings):
            self.index_to_row_id[embedding.id] = (
                self.embeddings.shape[0] - len(embeddings) + i
            )

        logger.info("Added embeddings to the embedding store.")

        if self.config.embedding_store.embedding_path is not None:
            embedding_store_path = (
                Path(self.config.dirs.data) / self.config.embedding_store.embedding_path
            )
            self.save(path=embedding_store_path)

    def reset(self) -> None:
        """This resets the embeddings store."""
        self.embeddings = np.zeros((0, self.embedding_dim))
        self.index_to_row_id = defaultdict()
        self.row_id_to_index
        logger.info("Reset the embeddings store.")

    def save(self, path: Path | str) -> None:
        """Save the embedding store to disk.

        Args:
            path:
                The path to the embeddings store in. This should be a .zip-file. This
                zip file will contain the embeddings matrix in the file
                `embeddings.npy` and the row ID to index mapping in the file
                `index_to_row_id.json`.

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

    def get_nearest_neighbours(self, embedding: np.ndarray) -> list[Index]:
        """Get the nearest neighbours to a given embedding.

        Args:
            embedding:
                The embedding to find nearest neighbours for.

        Returns:
            A list of indices of the nearest neighbours.

        Raises:
            ValueError:
                If the number of documents in the store is less than the number of
                documents to retrieve.
        """
        num_docs = self.config.embedding_store.num_documents_to_retrieve
        if self.embeddings.shape[0] < num_docs:
            raise ValueError(
                "The number of documents in the store is less than the number of "
                "documents to retrieve."
            )

        logger.info(f"Finding {num_docs:,} nearest neighbours...")
        scores = self.embeddings @ embedding
        top_indices = np.argsort(scores)[::-1][:num_docs]
        nearest_neighbours = [self.row_id_to_index[i] for i in top_indices]
        logger.info(f"Found nearest neighbours with indices {top_indices}.")
        return nearest_neighbours
