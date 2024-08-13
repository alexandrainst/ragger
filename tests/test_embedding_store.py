"""Unit tests for the `embedding_store` module."""

import typing

import numpy as np
import pytest
import ragger.embedding_store
from ragger.data_models import Embedding
from ragger.embedding_store import EmbeddingStore


@pytest.fixture(scope="module")
def embeddings(default_embedder) -> typing.Generator[list[Embedding], None, None]:
    """Initialise a list of documents for testing."""
    yield [
        Embedding(
            id="an id", embedding=np.ones(shape=(default_embedder.embedding_dim,))
        ),
        Embedding(
            id="another id", embedding=np.zeros(shape=(default_embedder.embedding_dim,))
        ),
    ]


@pytest.fixture(
    scope="module",
    params=[
        cls
        for cls in vars(ragger.embedding_store).values()
        if isinstance(cls, type)
        and issubclass(cls, EmbeddingStore)
        and cls is not EmbeddingStore
    ],
)
def embedding_store_cls(
    request,
) -> typing.Generator[typing.Type[EmbeddingStore], None, None]:
    """Initialise an embedding store class for testing."""
    yield request.param


@pytest.fixture(scope="module")
def embedding_store(
    embedding_store_cls, special_kwargs
) -> typing.Generator[EmbeddingStore, None, None]:
    """Initialise an embedding store for testing."""
    embedding_store = embedding_store_cls(
        **special_kwargs.get(embedding_store_cls.__name__, {})
    )
    yield embedding_store
    embedding_store.remove()


def test_initialisation(embedding_store):
    """Test that the embedding store can be initialised."""
    assert isinstance(embedding_store, EmbeddingStore)


def test_get_nearest_neighbours(embedding_store, embeddings):
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


def test_clear(embedding_store, embeddings):
    """Test that the embedding store can be cleared."""
    embedding_store.add_embeddings(embeddings=embeddings)
    embedding_store.clear()
    assert len(embedding_store) == 0


def test_getitem(embedding_store, embeddings):
    """Test that embeddings can be fetched from the embedding store."""
    embedding_store.clear()
    embedding_store.add_embeddings(embeddings=embeddings)
    for embedding in embeddings:
        assert embedding_store[embedding.id] == embedding


def test_getitem_missing(embedding_store, embeddings, non_existing_id):
    """Test that fetching a missing embedding raises a KeyError."""
    embedding_store.clear()
    embedding_store.add_embeddings(embeddings=embeddings)
    with pytest.raises(KeyError):
        embedding_store[non_existing_id]


def test_contains(embeddings, embedding_store, non_existing_id):
    """Test that the embedding store can check if it contains a embedding."""
    embedding_store.clear()
    embedding_store.add_embeddings(embeddings=embeddings)
    for embedding in embeddings:
        assert embedding.id in embedding_store
    assert non_existing_id not in embedding_store


def test_len(embedding_store, embeddings):
    """Test that the embedding store can return the number of embeddings."""
    embedding_store.clear()
    embedding_store.add_embeddings(embeddings=embeddings)
    assert len(embedding_store) == len(embeddings)
