"""Store and fetch embeddings from a database."""

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from transformers import AutoConfig

from .utils import Embedding, Index


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
        self._embeddings = np.zeros((0, self.embedding_dim))
        self._index_to_row_id: dict[Index, int] = defaultdict()
        self._row_id_to_index: dict[int, Index] = defaultdict()
        self.embeddings: list[Embedding] = list()

    def _get_embedding_dimension(self) -> int:
        """This returns the embedding dimension for the embedding model.

        Returns:
            The embedding dimension.
        """
        model_config = AutoConfig.from_pretrained(self.config.embedder.e5.model_id)
        return model_config.hidden_size

    def add_embeddings(self, embeddings: list[Embedding]) -> None:
        """Add embeddings to the store.

        Args:
            embeddings:
                A list of embeddings to add to the store.
        """
        for embedding in embeddings:
            if embedding.id in self._index_to_row_id:
                # Update the embedding at the corresponding row, if index is in the
                # index to row id dictionary.
                self._embeddings[self._index_to_row_id[embedding.id], :] = (
                    embedding.embedding
                )

                # Update the embedding in the embeddings list.
                self.embeddings[self._index_to_row_id[embedding.id]] = embedding
            else:
                # Add the embedding to the embeddings array and update the index to row
                # id dictionary.
                self._embeddings = np.vstack(
                    [self._embeddings, np.array(embedding.embedding)]
                )
                self._index_to_row_id[embedding.id] = len(self._embeddings)
                self._row_id_to_index[len(self._embeddings)] = embedding.id
                self.embeddings.append(embedding)

    def reset(self) -> None:
        """This resets the embeddings store."""
        self._embeddings = np.zeros((0, self.embedding_dim))
        self._index_to_row_id = defaultdict()
        self._row_id_to_index = defaultdict()
        self.embeddings = list()

    def save(self, path: Path | str) -> None:
        """This saves the embeddings store to disk.

        This will store the embeddings in `npy`-file, called
        `embeddings.npy`.

        Args:
            path:
                The path to the embeddings store in.
        """
        path = Path(path)
        np.savez_compressed(file=path, _embeddings=self._embeddings)

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

        Raises:
            ValueError:
                If the number of documents in the store is less than the number of
                documents to retrieve.
        """
        num_docs = self.config.embedding_store.numpy.num_documents_to_retrieve
        if self._embeddings.shape[0] < num_docs:
            raise ValueError(
                "The number of documents in the store is less than the number of "
                "documents to retrieve."
            )
        scores = self._embeddings @ embedding
        top_indices = np.argsort(scores)[::-1][:num_docs]
        nearest_neighbours = [self._row_id_to_index[i] for i in top_indices.to_list()]
        return nearest_neighbours
