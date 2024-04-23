"""Store and fetch embeddings from a database."""

import io
import json
import zipfile
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
            else:
                # Add the embedding to the embeddings array and update the index to row
                # id dictionary.
                self._embeddings = np.vstack(
                    [self._embeddings, np.array(embedding.embedding)]
                )
                self._index_to_row_id[embedding.id] = len(self._embeddings)
                self._row_id_to_index[len(self._embeddings)] = embedding.id

    def reset(self) -> None:
        """This resets the embeddings store."""
        self._embeddings = np.zeros((0, self.embedding_dim))
        self._index_to_row_id = defaultdict()
        self._row_id_to_index = defaultdict()

    def save(self, path: Path | str) -> None:
        """This saves the embeddings store to disk.

        This will store the embeddings in `npy`-file, called
        `embeddings.npy`.

        Args:
            path:
                The path to the embeddings store in.
        """
        path = Path(path)
        array_file = io.BytesIO()
        np.save(file=array_file, _embeddings=self._embeddings)

        index_to_row_id = json.dumps(self._index_to_row_id).encode("utf-8")
        row_id_to_index = json.dumps(self._row_id_to_index).encode("utf-8")

        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("embeddings.npy", data=array_file.getvalue())
            zf.writestr("index_to_row_id.json", data=index_to_row_id)
            zf.writestr("row_id_to_index.json", data=row_id_to_index)

    def load(self, path: Path | str) -> None:
        """This loads the embeddings store from disk.

        Args:
            path:
                The path to the zip file to load the embeddings store from.
        """
        path = Path(path)
        with zipfile.ZipFile(file=path, mode="r") as zf:
            index_to_row_id_encoded = zf.read("index_to_row_id.json")
            index_to_row_id = json.loads(index_to_row_id_encoded.decode("utf-8"))
            row_id_to_index_encoded = zf.read("row_id_to_index.json")
            row_id_to_index = json.loads(row_id_to_index_encoded.decode("utf-8"))
            array_file = io.BytesIO(zf.read("embeddings.npy"))
            embeddings = np.load(file=array_file, allow_pickle=False)
        self._embeddings = embeddings
        self._index_to_row_id = index_to_row_id
        self._row_id_to_index = row_id_to_index

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
