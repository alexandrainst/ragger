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
def document_store(documents, request) -> typing.Generator[DocumentStore, None, None]:
    """Initialise a document store for testing."""
    document_store = request.param()
    document_store.add_documents(documents=documents)
    yield document_store
    document_store.remove()


def test_initialisation(document_store):
    """Test that the document store can be initialised."""
    assert isinstance(document_store, DocumentStore)


def test_getitem(document_store):
    """Test that documents can be fetched from the document store."""
    assert document_store["1"].text == "Den hvide kat hedder Sjusk."
    assert document_store["2"].text == "Den sorte kat hedder Sutsko."
    assert document_store["3"].text == "Den røde kat hedder Pjuskebusk."
    assert document_store["4"].text == "Den grønne kat hedder Sjask."
    assert document_store["5"].text == "Den blå kat hedder Sky."


def test_getitem_missing(document_store):
    """Test that fetching a missing document raises a KeyError."""
    with pytest.raises(KeyError):
        document_store["6"]
