"""Test fixtures used throughout the test suite."""

import typing
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from ragger.data_models import Document, DocumentStore, Generator, Retriever
from ragger.document_store import JsonlDocumentStore
from ragger.generator import OpenAIGenerator
from ragger.rag_system import RagSystem
from ragger.retriever import EmbeddingRetriever


@pytest.fixture(scope="session")
def special_kwargs() -> typing.Generator[dict[str, dict[str, typing.Any]], None, None]:
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
def default_retriever() -> typing.Generator[Retriever, None, None]:
    """A retriever for testing."""
    yield EmbeddingRetriever()


@pytest.fixture(scope="session")
def default_generator() -> typing.Generator[Generator, None, None]:
    """A generator for testing."""
    yield OpenAIGenerator()


@pytest.fixture(scope="session")
def rag_system(
    default_document_store, default_retriever, default_generator
) -> typing.Generator[RagSystem, None, None]:
    """A RAG system for testing."""
    yield RagSystem(
        document_store=default_document_store,
        retriever=default_retriever,
        generator=default_generator,
    )
