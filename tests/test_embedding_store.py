"""Unit tests for the `embedding_store` module."""

import typing
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import ragger.embedding_store
from ragger.data_models import Embedding
from ragger.embedding_store import EmbeddingStore, NumpyEmbeddingStore


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
    embedding_store_cls,
) -> typing.Generator[EmbeddingStore, None, None]:
    """Initialise an embedding store for testing."""
    embedding_store = embedding_store_cls()
    yield embedding_store
    embedding_store.remove()


@pytest.fixture(scope="module")
def embeddings() -> typing.Generator[list[Embedding], None, None]:
    """Initialise a list of documents for testing."""
    yield [
        Embedding(id="an id", embedding=np.ones(shape=(8,))),
        Embedding(id="another id", embedding=np.zeros(shape=(8,))),
    ]


def test_initialisation(embedding_store):
    """Test that the embedding store can be initialised."""
    assert isinstance(embedding_store, EmbeddingStore)


def test_add_embeddings(embedding_store, embeddings):
    """Test that embeddings can be added to the embedding store."""
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
    assert embedding_store.embeddings.shape == (0, embedding_store.embedding_dim)


def test_save_load(embedding_store, embeddings):
    """Test that the embedding store can be saved."""
    embedding_store.clear()
    embedding_store.add_embeddings(embeddings=embeddings)
    new_store = NumpyEmbeddingStore(embedding_dim=embedding_store.embedding_dim)
    with NamedTemporaryFile(suffix=".zip") as file:
        embedding_store.save(file.name)
        new_store.load(file.name)
        assert np.array_equal(new_store.embeddings, embedding_store.embeddings)
        assert new_store.embedding_dim == embedding_store.embedding_dim
