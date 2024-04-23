"""Unit tests for the `embedder` module."""

from typing import Generator

import numpy as np
import pytest
from omegaconf import DictConfig
from ragger.embedder import E5Embedder, Embedder
from ragger.utils import Document


class TestE5Embedder:
    """Tests for the `Embedder` class."""

    @pytest.fixture(scope="class")
    def embedder(self) -> Generator[E5Embedder, None, None]:
        """Initialise an Embedder for testing."""
        config = DictConfig(dict(embedder_id="intfloat/multilingual-e5-large"))
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
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(documents)

    def test_embed_query(self, embedder, query):
        """Test that the Embedder can embed a query."""
        embeddings = embedder.embed_query(query)
        assert isinstance(embeddings, np.ndarray)

    def test_prepare_query(self, embedder, query):
        """Test that the Embedder can prepare a query."""
        prepared_query = embedder._prepare_query_for_embedding(query)
        assert isinstance(prepared_query, str)
        assert "?" in prepared_query

    def test_prepare_texts(self, embedder, documents):
        """Test that the Embedder can prepare texts for embedding."""
        texts = [document.text for document in documents]
        prepared_texts = embedder._prepare_texts_for_embedding(texts)
        assert isinstance(prepared_texts, list)
        assert len(prepared_texts) == len(texts)
