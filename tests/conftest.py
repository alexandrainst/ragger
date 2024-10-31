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
from ragger.embedder import OpenAIEmbedder
from ragger.embedding_store import NumpyEmbeddingStore
from ragger.generator import OpenAIGenerator
from ragger.rag_system import RagSystem


@pytest.fixture(scope="session")
def special_kwargs() -> typing.Generator[dict[str, dict[str, str]], None, None]:
    """Special keyword arguments for initialising RAG components."""
    yield dict(
        E5Embedder=dict(embedder_model_id="intfloat/multilingual-e5-small"),
        VllmGenerator=dict(model_id="mhenrichsen/danskgpt-tiny-chat"),
        GGUFGenerator=dict(model_id="hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF"),
    )


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
def query() -> typing.Generator[str, None, None]:
    """Initialise a query for testing."""
    yield "Hvad hedder den hvide kat?"


@pytest.fixture(scope="session")
def non_existing_id() -> typing.Generator[str, None, None]:
    """Initialise a non-existing ID for testing."""
    yield "non-existing-id"


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
    yield OpenAIEmbedder()


@pytest.fixture(scope="session")
def default_embedding_store() -> typing.Generator[EmbeddingStore, None, None]:
    """An embedding store for testing."""
    embedding_store = NumpyEmbeddingStore()
    yield embedding_store
    embedding_store.clear()


@pytest.fixture(scope="session")
def default_generator() -> typing.Generator[Generator, None, None]:
    """A generator for testing."""
    yield OpenAIGenerator()


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
