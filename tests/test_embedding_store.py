"""Unit tests for the `embedding_store` module."""

from tempfile import NamedTemporaryFile
from typing import Generator

import numpy as np
import pytest
from omegaconf import DictConfig
from ragger.embedding_store import EmbeddingStore, NumpyEmbeddingStore


class TestNumpyEmbeddingStore:
    """Tests for the `NumpyEmbeddingStore` class."""

    @pytest.fixture(scope="class")
    def embedding_store(self) -> Generator[NumpyEmbeddingStore, None, None]:
        """Initialise a NumpyEmbeddingStore for testing."""
        config = DictConfig(
            dict(
                num_documents_to_retrieve=2,
                embedder_id="intfloat/multilingual-e5-large",
            )
        )
        store = NumpyEmbeddingStore(config=config)
        yield store

    @pytest.fixture(scope="class")
    def embeddings(self, embedding_store) -> list[np.array]:
        """Initialise a list of documents for testing."""
        return [
            np.ones(shape=(embedding_store.embedding_dim,)),
            np.zeros(shape=(embedding_store.embedding_dim,)),
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
        assert np.array_equal(embedding_store.embeddings[0], embeddings[0])
        assert np.array_equal(embedding_store.embeddings[1], embeddings[1])
        embedding_store.reset()

    def test_get_nearest_neighbours(self, embedding_store, embeddings):
        """Test that the nearest neighbours to an embedding can be found."""
        embedding_store.add_embeddings(embeddings)
        neighbours = embedding_store.get_nearest_neighbours(embeddings[0])
        assert np.array_equal(np.array(neighbours), np.array([0, 1]))
        neighbours = embedding_store.get_nearest_neighbours(embeddings[1])
        assert np.array_equal(np.array(neighbours), np.array([1, 0]))
        embedding_store.reset()

    def test_reset(self, embedding_store, embeddings):
        """Test that the NumpyEmbeddingStore can be reset."""
        embedding_store.add_embeddings(embeddings)
        embedding_store.reset()
        assert embedding_store.embeddings.shape == (0, embedding_store.embedding_dim)
        embedding_store.reset()

    def test_save_load(self, embedding_store, embeddings):
        """Test that the NumpyEmbeddingStore can be saved."""
        embedding_store.add_embeddings(embeddings)
        new_store = NumpyEmbeddingStore(embedding_store.config)
        with NamedTemporaryFile(suffix=".npy") as file:
            embedding_store.save(file.name)
            new_store.load(file.name)
            assert np.array_equal(new_store.embeddings, embedding_store.embeddings)
            assert new_store.embedding_dim == embedding_store.embedding_dim
        embedding_store.reset()