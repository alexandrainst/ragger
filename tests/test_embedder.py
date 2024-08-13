"""Unit tests for the `embedder` module."""

import typing

import numpy as np
import pytest
import ragger.embedder
from ragger.data_models import Embedding
from ragger.embedder import Embedder


@pytest.fixture(
    scope="module",
    params=[
        cls
        for cls in vars(ragger.embedder).values()
        if isinstance(cls, type) and issubclass(cls, Embedder) and cls is not Embedder
    ],
)
def embedder(request) -> typing.Generator[Embedder, None, None]:
    """Initialise an embedder for testing."""
    embedder = request.param()
    yield embedder


def test_initialisation(embedder):
    """Test that the embedder can be initialised."""
    assert isinstance(embedder, Embedder)


def test_embed_documents(embedder, documents):
    """Test that the embedder can embed text."""
    embeddings = embedder.embed_documents(documents)
    assert isinstance(embeddings, list)
    for embedding in embeddings:
        assert isinstance(embedding, Embedding)
    assert len(embeddings) == len(documents)


def test_embed_query(embedder, query):
    """Test that the embedder can embed a query."""
    embeddings = embedder.embed_query(query)
    assert isinstance(embeddings, np.ndarray)


def test_max_context_length(embedder):
    """Test that the embedder can return the maximum context length."""
    assert isinstance(embedder.max_context_length, int)
    assert embedder.max_context_length > 0
