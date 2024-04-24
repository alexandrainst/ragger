"""Unit tests for the `embedding_store` module."""

from tempfile import NamedTemporaryFile
from typing import Generator

import numpy as np
import pytest
from omegaconf import DictConfig
from ragger.embedding_store import EmbeddingStore, NumpyEmbeddingStore
from ragger.utils import Embedding


class TestNumpyEmbeddingStore:
    """Tests for the `NumpyEmbeddingStore` class."""

    @pytest.fixture(scope="class")
    def embedding_store(self) -> Generator[NumpyEmbeddingStore, None, None]:
        """Initialise a NumpyEmbeddingStore for testing."""
        config = DictConfig(
            dict(
                embedding_store=dict(numpy=dict(num_documents_to_retrieve=2)),
                embedder=dict(
                    e5=dict(
                        model_id="intfloat/multilingual-e5-small",
                        document_text_field="text",
                    )
                ),
            )
        )
        store = NumpyEmbeddingStore(config=config)
        yield store

    @pytest.fixture(scope="class")
    def embeddings(self, embedding_store) -> list[np.ndarray]:
        """Initialise a list of documents for testing."""
        return [
            Embedding(
                id="a id", embedding=np.ones(shape=(embedding_store.embedding_dim,))
            ),
            Embedding(
                id="another id",
                embedding=np.zeros(shape=(embedding_store.embedding_dim,)),
            ),
        ]

    def is_embedding_store(self):
        """Test that the NumpyEmbeddingStore is an EmbeddingStore."""
        assert issubclass(NumpyEmbeddingStore, EmbeddingStore)

    def test_initialisation(self, embedding_store):
        """Test that the NumpyEmbeddingStore can be initialised."""
        assert embedding_store

    def test_add_embeddings(self, embedding_store, embeddings):
        """Test that embeddings can be added to the NumpyEmbeddingStore."""
        embedding_store.add_embeddings(embeddings)
        assert len(embedding_store.embeddings) == 2
        assert np.array_equal(
            embedding_store.embeddings[embedding_store.index_to_row_id["a id"]],
            embeddings[0].embedding,
        )
        assert np.array_equal(
            embedding_store.embeddings[embedding_store.index_to_row_id["another id"]],
            embeddings[1].embedding,
        )
        embedding_store.reset()

    def test_get_nearest_neighbours(self, embedding_store, embeddings):
        """Test that the nearest neighbours to an embedding can be found."""
        embedding_store.add_embeddings(embeddings)
        neighbours = embedding_store.get_nearest_neighbours(embeddings[0].embedding)
        assert neighbours == ["a id", "another id"]
        neighbours = embedding_store.get_nearest_neighbours(embeddings[1].embedding)
        assert neighbours == ["another id", "a id"]
        embedding_store.reset()

    def test_reset(self, embedding_store, embeddings):
        """Test that the NumpyEmbeddingStore can be reset."""
        embedding_store.add_embeddings(embeddings)
        embedding_store.reset()
        assert embedding_store.embeddings.shape == (0, embedding_store.embedding_dim)
        embedding_store.reset()

    def test_indices_already_in_store(self, embedding_store, embeddings):
        """Test that an error is raised when indices are already in the store."""
        embedding_store.add_embeddings(embeddings)
        with pytest.raises(ValueError):
            embedding_store.add_embeddings(embeddings)
        embedding_store.reset()

    def test_save_load(self, embedding_store, embeddings):
        """Test that the NumpyEmbeddingStore can be saved."""
        embedding_store.add_embeddings(embeddings)
        new_store = NumpyEmbeddingStore(embedding_store.config)
        with NamedTemporaryFile(suffix=".zip") as file:
            embedding_store.save(file.name)
            new_store.load(file.name)
            assert np.array_equal(new_store.embeddings, embedding_store.embeddings)
            assert new_store.embedding_dim == embedding_store.embedding_dim
        embedding_store.reset()
