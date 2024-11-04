"""Unit tests for the `embedding_store` module."""

import typing

import numpy as np
import pytest

import ragger.embedding_store
from ragger.data_models import Embedder, Embedding
from ragger.embedder import E5Embedder
from ragger.embedding_store import EmbeddingStore, PostgresEmbeddingStore


@pytest.fixture(scope="module")
def default_embedder(special_kwargs) -> typing.Generator[Embedder, None, None]:
    """Initialise the default embedder for testing."""
    yield E5Embedder(**special_kwargs.get("E5Embedder", {}))


@pytest.fixture(scope="module")
def embeddings(default_embedder) -> typing.Generator[list[Embedding], None, None]:
    """Initialise a list of documents for testing."""
    rng = np.random.default_rng(seed=4242)
    yield [
        Embedding(
            id="an id", embedding=rng.random(size=(default_embedder.embedding_dim,))
        ),
        Embedding(
            id="another id",
            embedding=rng.random(size=(default_embedder.embedding_dim,)),
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
    embedding_store: EmbeddingStore = embedding_store_cls(
        **special_kwargs.get(embedding_store_cls.__name__, {})
    )
    yield embedding_store
    embedding_store.remove()
    drop_table_if_postgres_embedding_store(embedding_store=embedding_store)


def drop_table_if_postgres_embedding_store(embedding_store: EmbeddingStore) -> None:
    """Drop the table if the embedding store is a PostgresEmbeddingStore."""
    if isinstance(embedding_store, PostgresEmbeddingStore):
        with embedding_store._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS embeddings")


def test_initialisation(embedding_store):
    """Test that the embedding store can be initialised."""
    assert isinstance(embedding_store, EmbeddingStore)


def test_get_nearest_neighbours(embedding_store, embeddings):
    """Test that the nearest neighbours to an embedding can be found."""
    embedding_store.remove()
    drop_table_if_postgres_embedding_store(embedding_store=embedding_store)
    embedding_store.add_embeddings(embeddings=embeddings)
    neighbours = embedding_store.get_nearest_neighbours(
        embedding=embeddings[0].embedding
    )
    assert neighbours == ["an id", "another id"]
    neighbours = embedding_store.get_nearest_neighbours(
        embedding=embeddings[1].embedding
    )
    assert neighbours == ["another id", "an id"]


def test_remove(embedding_store, embeddings):
    """Test that the embedding store can be removed."""
    embedding_store.add_embeddings(embeddings=embeddings)
    embedding_store.remove()
    drop_table_if_postgres_embedding_store(embedding_store=embedding_store)
    assert len(embedding_store) == 0


def test_getitem(embedding_store, embeddings):
    """Test that embeddings can be fetched from the embedding store."""
    embedding_store.remove()
    drop_table_if_postgres_embedding_store(embedding_store=embedding_store)
    embedding_store.add_embeddings(embeddings=embeddings)
    for embedding in embeddings:
        assert embedding_store[embedding.id] == embedding


def test_getitem_missing(embedding_store, embeddings, non_existing_id):
    """Test that fetching a missing embedding raises a KeyError."""
    embedding_store.remove()
    drop_table_if_postgres_embedding_store(embedding_store=embedding_store)
    embedding_store.add_embeddings(embeddings=embeddings)
    with pytest.raises(KeyError):
        embedding_store[non_existing_id]


def test_contains(embeddings, embedding_store, non_existing_id):
    """Test that the embedding store can check if it contains a embedding."""
    embedding_store.remove()
    drop_table_if_postgres_embedding_store(embedding_store=embedding_store)
    embedding_store.add_embeddings(embeddings=embeddings)
    for embedding in embeddings:
        assert embedding.id in embedding_store
    assert non_existing_id not in embedding_store


def test_len(embedding_store, embeddings):
    """Test that the embedding store can return the number of embeddings."""
    embedding_store.remove()
    drop_table_if_postgres_embedding_store(embedding_store=embedding_store)
    embedding_store.add_embeddings(embeddings=embeddings)
    assert len(embedding_store) == len(embeddings)
