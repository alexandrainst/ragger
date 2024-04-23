"""Store and fetch embeddings from a database."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from transformers import AutoConfig

from .utils import Index


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
    def add_embeddings(self, embeddings: list[np.ndarray]) -> None:
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
        self.embeddings = np.empty(shape=(self.embedding_dim,))

    def _get_embedding_dimension(self) -> int:
        """This returns the embedding dimension for the embedding model.

        Returns:
            The embedding dimension.
        """
        model_config = AutoConfig.from_pretrained(self.config.embedder_id)
        return model_config.hidden_size

    def add_embeddings(self, embeddings: list[np.ndarray]) -> None:
        """Add embeddings to the store.

        Args:
            embeddings:
                A list of embeddings to add to the store.
        """
        self.embeddings = np.vstack([self.embeddings, np.array(embeddings)])

    def reset(self) -> None:
        """This resets the embeddings store."""
        self.embeddings = np.empty(shape=(self.embedding_dim,))

    def save(self, path: Path | str) -> None:
        """This saves the embeddings store to disk.

        This will store the embeddings in `npy`-file, called
        `embeddings.npy`.

        Args:
            path: The path to the embeddings store in.
        """
        path = Path(path)
        np.save(file=path, arr=self.embeddings)

    def load(self, path: Path | str) -> None:
        """This loads the embeddings store from disk.

        Args:
            path:
                The path to the zip file to load the embeddings store from.
        """
        path = Path(path)
        embeddings = np.load(file=path, allow_pickle=False)
        assert self.embedding_dim == embeddings.shape[1]
        self.embeddings = embeddings

    def get_nearest_neighbours(self, embedding: np.ndarray) -> list[Index]:
        """Get the nearest neighbours to a given embedding.

        Args:
            embedding:
                The embedding to find nearest neighbours for.

        Returns:
            A list of indices of the nearest neighbours.
        """
        # Get the top-k documents
        num_documents = self.config.num_documents_to_retrieve
        scores = self.embeddings @ embedding
        top_indices = np.argsort(scores)[::-1][:num_documents]
        return top_indices
