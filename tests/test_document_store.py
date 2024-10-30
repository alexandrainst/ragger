"""Unit tests for the `document_store` module."""

import typing

import pytest

import ragger.document_store
from ragger.data_models import DocumentStore


@pytest.fixture(
    scope="module",
    params=[
        cls
        for cls in vars(ragger.document_store).values()
        if isinstance(cls, type)
        and issubclass(cls, DocumentStore)
        and cls is not DocumentStore
    ],
)
def document_store(
    documents, request, special_kwargs, rag_system
) -> typing.Generator[DocumentStore, None, None]:
    """Initialise a document store for testing."""
    document_store_cls = request.param
    document_store: DocumentStore = document_store_cls(
        **special_kwargs.get(document_store_cls.__name__, {})
    )
    document_store.compile(
        embedder=rag_system.embedder,
        embedding_store=rag_system.embedding_store,
        generator=rag_system.generator,
    )
    document_store.add_documents(documents=documents)
    yield document_store
    document_store.remove()


def test_initialisation(document_store):
    """Test that the document store can be initialised."""
    assert isinstance(document_store, DocumentStore)


def test_getitem(document_store, documents):
    """Test that documents can be fetched from the document store."""
    for document in documents:
        assert document_store[document.id] == document


def test_getitem_missing(document_store, non_existing_id):
    """Test that fetching a missing document raises a KeyError."""
    with pytest.raises(KeyError):
        document_store[non_existing_id]


def test_contains(documents, document_store, non_existing_id):
    """Test that the document store can check if it contains a document."""
    for document in documents:
        assert document.id in document_store
    assert non_existing_id not in document_store


def test_len(document_store, documents):
    """Test that the document store can return the number of documents."""
    assert len(document_store) == len(documents)
