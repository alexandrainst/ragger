"""Unit tests for the `embedding_store` module."""

import typing
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from ragger.data_models import Embedding
from ragger.embedding_store import EmbeddingStore, NumpyEmbeddingStore


class TestNumpyEmbeddingStore:
    """Tests for the `NumpyEmbeddingStore` class."""

    @pytest.fixture(scope="class")
    def embedding_store(
        self, default_embedder
    ) -> typing.Generator[NumpyEmbeddingStore, None, None]:
        """Initialise a NumpyEmbeddingStore for testing."""
        embedding_store = NumpyEmbeddingStore(
            embedding_dim=default_embedder.embedding_dim,
            path=Path("test-embeddings.zip"),
        )
        yield embedding_store
        embedding_store.clear()

    @pytest.fixture(scope="class")
    def embeddings(
        self, embedding_store
    ) -> typing.Generator[list[Embedding], None, None]:
        """Initialise a list of documents for testing."""
        yield [
            Embedding(
                id="an id", embedding=np.ones(shape=(embedding_store.embedding_dim,))
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
        assert isinstance(embedding_store, NumpyEmbeddingStore)

    def test_add_embeddings(self, embedding_store, embeddings):
        """Test that embeddings can be added to the NumpyEmbeddingStore."""
        embedding_store.clear()
        embedding_store.add_embeddings(embeddings=embeddings)
        assert len(embedding_store.embeddings) == 2
        assert np.array_equal(
            embedding_store.embeddings[embedding_store.index_to_row_id["an id"]],
            embeddings[0].embedding,
        )
        assert np.array_equal(
            embedding_store.embeddings[embedding_store.index_to_row_id["another id"]],
            embeddings[1].embedding,
        )

    def test_get_nearest_neighbours(self, embedding_store, embeddings):
        """Test that the nearest neighbours to an embedding can be found."""
        embedding_store.clear()
        embedding_store.add_embeddings(embeddings=embeddings)
        neighbours = embedding_store.get_nearest_neighbours(
            embedding=embeddings[0].embedding
        )
        assert neighbours == ["an id", "another id"]
        neighbours = embedding_store.get_nearest_neighbours(
            embedding=embeddings[1].embedding
        )
        assert neighbours == ["another id", "an id"]

    def test_clear(self, embedding_store, embeddings):
        """Test that the NumpyEmbeddingStore can be cleared."""
        embedding_store.add_embeddings(embeddings=embeddings)
        embedding_store.clear()
        assert embedding_store.embeddings.shape == (0, embedding_store.embedding_dim)

    def test_save_load(self, embedding_store, embeddings):
        """Test that the NumpyEmbeddingStore can be saved."""
        embedding_store.clear()
        embedding_store.add_embeddings(embeddings=embeddings)
        new_store = NumpyEmbeddingStore(embedding_dim=embedding_store.embedding_dim)
        with NamedTemporaryFile(suffix=".zip") as file:
            embedding_store.save(file.name)
            new_store.load(file.name)
            assert np.array_equal(new_store.embeddings, embedding_store.embeddings)
            assert new_store.embedding_dim == embedding_store.embedding_dim
