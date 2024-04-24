"""Unit tests for the `document_store` module."""

from typing import Generator

import pytest
from ragger.document_store import DocumentStore, JsonlDocumentStore


class TestJsonlDocumentStore:
    """Tests for the `JsonlDocumentStore` class."""

    @pytest.fixture(scope="class")
    def document_store(self, valid_config) -> Generator[JsonlDocumentStore, None, None]:
        """Initialise a JsonlDocumentStore for testing."""
        store = JsonlDocumentStore(config=valid_config)
        yield store

    def test_is_document_store(self):
        """Test that the JsonlDocumentStore is a DocumentStore."""
        assert issubclass(JsonlDocumentStore, DocumentStore)

    def test_initialisation(self, document_store):
        """Test that the JsonlDocumentStore can be initialised."""
        assert document_store

    def test_getitem(self, document_store):
        """Test that documents can be fetched from the JsonlDocumentStore."""
        assert document_store["1"].text == "Den hvide kat hedder Sjusk."
        assert document_store["2"].text == "Den sorte kat hedder Sutsko."
        assert document_store["3"].text == "Den røde kat hedder Pjuskebusk."
        assert document_store["4"].text == "Den grønne kat hedder Sjask."
        assert document_store["5"].text == "Den blå kat hedder Sky."

    def test_getitem_missing(self, document_store):
        """Test that fetching a missing document raises a KeyError."""
        with pytest.raises(KeyError):
            document_store["6"]
