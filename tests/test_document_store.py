"""Unit tests for the `document_store` module."""

import json
from tempfile import NamedTemporaryFile
from typing import Generator

import pytest
from omegaconf import DictConfig
from ragger.document_store import DocumentStore, JsonlDocumentStore


class TestJsonlDocumentStore:
    """Tests for the `JsonlDocumentStore` class."""

    @pytest.fixture(scope="class")
    def document_store(self) -> Generator[JsonlDocumentStore, None, None]:
        """Initialise a JsonlDocumentStore for testing."""
        with NamedTemporaryFile(mode="w", suffix=".jsonl") as file:
            # Create a JSONL file with some documents
            data_dicts = [
                dict(id="1", text="Hello, world!"),
                dict(id="2", text="Goodbye, world!"),
            ]
            data_str = "\n".join(json.dumps(data_dict) for data_dict in data_dicts)
            file.write(data_str)
            file.flush()

            # Create a JsonlDocumentStore using the temporary file
            config = DictConfig(
                dict(document_store=dict(jsonl=dict(filename=file.name)))
            )
            store = JsonlDocumentStore(config=config)
            yield store
        del store

    def test_is_document_store(self):
        """Test that the JsonlDocumentStore is a DocumentStore."""
        assert issubclass(JsonlDocumentStore, DocumentStore)

    def test_initialisation(self, document_store):
        """Test that the JsonlDocumentStore can be initialised."""
        assert document_store

    def test_getitem(self, document_store):
        """Test that documents can be fetched from the JsonlDocumentStore."""
        assert document_store["1"].text == "Hello, world!"
        assert document_store["2"].text == "Goodbye, world!"

    def test_getitem_missing(self, document_store):
        """Test that fetching a missing document raises a KeyError."""
        with pytest.raises(KeyError):
            document_store["3"]
