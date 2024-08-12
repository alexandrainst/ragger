"""Test fixtures used throughout the test suite."""

import typing
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from ragger.data_models import (
    Document,
    DocumentStore,
    Embedder,
    EmbeddingStore,
    Generator,
)
from ragger.document_store import JsonlDocumentStore
from ragger.embedder import E5Embedder
from ragger.embedding_store import NumpyEmbeddingStore
from ragger.generator import OpenaiGenerator
from ragger.rag_system import RagSystem


@pytest.fixture(scope="session")
def documents() -> typing.Generator[list[Document], None, None]:
    """Some documents for testing."""
    yield [
        Document(id="1", text="Den hvide kat hedder Sjusk."),
        Document(id="2", text="Den sorte kat hedder Sutsko."),
        Document(id="3", text="Den røde kat hedder Pjuskebusk."),
        Document(id="4", text="Den grønne kat hedder Sjask."),
        Document(id="5", text="Den blå kat hedder Sky."),
    ]


@pytest.fixture(scope="session")
def default_document_store(documents) -> typing.Generator[DocumentStore, None, None]:
    """A document store for testing."""
    with NamedTemporaryFile(mode="w", suffix=".jsonl") as file:
        data_str = "\n".join(document.model_dump_json() for document in documents)
        file.write(data_str)
        file.flush()
        yield JsonlDocumentStore(path=Path(file.name))
        file.close()


@pytest.fixture(scope="session")
def default_embedder() -> typing.Generator[Embedder, None, None]:
    """An embedder for testing."""
    yield E5Embedder(embedder_model_id="intfloat/multilingual-e5-small")


@pytest.fixture(scope="session")
def default_embedding_store(
    default_embedder,
) -> typing.Generator[EmbeddingStore, None, None]:
    """An embedding store for testing."""
    embedding_store = NumpyEmbeddingStore(
        embedding_dim=default_embedder.embedding_dim, path=Path("test-embeddings.zip")
    )
    yield embedding_store
    embedding_store.clear()


@pytest.fixture(scope="session")
def default_generator() -> typing.Generator[Generator, None, None]:
    """A generator for testing."""
    yield OpenaiGenerator()


@pytest.fixture(scope="session")
def rag_system(
    default_document_store, default_embedder, default_embedding_store, default_generator
) -> typing.Generator[RagSystem, None, None]:
    """A RAG system for testing."""
    yield RagSystem(
        document_store=default_document_store,
        embedder=default_embedder,
        embedding_store=default_embedding_store,
        generator=default_generator,
    )
