"""Unit tests for the `embedder` module."""

import typing

import numpy as np
import pytest
from omegaconf import DictConfig
from ragger.data_models import Document, Embedding
from ragger.embedder import E5Embedder, Embedder


class TestE5Embedder:
    """Tests for the `Embedder` class."""

    @pytest.fixture(scope="class")
    def config(
        self, dirs_params, embedder_params
    ) -> typing.Generator[DictConfig, None, None]:
        """Initialise a configuration for testing."""
        yield DictConfig(
            dict(dirs=dirs_params, embedder=embedder_params, verbose=False)
        )

    @pytest.fixture(scope="class")
    def embedder(self, config) -> typing.Generator[E5Embedder, None, None]:
        """Initialise an Embedder for testing."""
        embedder = E5Embedder(config=config)
        yield embedder

    @pytest.fixture(scope="class")
    def documents(self) -> list[Document]:
        """Initialise a list of documents for testing."""
        return [
            Document(id="1", text="Hello, world!"),
            Document(id="2", text="Goodbye, world!"),
        ]

    @pytest.fixture(scope="class")
    def query(self) -> str:
        """Initialise a query for testing."""
        return "Hello, world!"

    def is_embedder(self):
        """Test that the Embedder is an ABC."""
        assert issubclass(E5Embedder, Embedder)

    def test_initialisation(self, embedder):
        """Test that the Embedder can be initialised."""
        assert embedder

    def test_embed(self, embedder, documents):
        """Test that the Embedder can embed text."""
        embeddings = embedder.embed_documents(documents)
        assert isinstance(embeddings, list)
        for embedding in embeddings:
            assert isinstance(embedding, Embedding)
        assert len(embeddings) == len(documents)

    def test_embed_query(self, embedder, query):
        """Test that the Embedder can embed a query."""
        embeddings = embedder.embed_query(query)
        assert isinstance(embeddings, np.ndarray)
