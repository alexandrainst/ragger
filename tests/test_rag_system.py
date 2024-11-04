"""Unit tests for the `rag_system` module."""

import json
import tempfile
import typing

import pytest

from ragger.data_models import Document
from ragger.rag_system import RagSystem


@pytest.fixture(scope="module")
def answer_and_documents(
    rag_system,
) -> typing.Generator[tuple[str, list[Document]], None, None]:
    """An answer and supporting documents for testing."""
    yield rag_system.answer("Hvad farve har Sutsko?")


def test_initialisation(rag_system):
    """Test that the RagSystem can be initialised."""
    assert rag_system


def test_answer_is_non_empty(answer_and_documents):
    """Test that the answer is non-empty."""
    answer, _ = answer_and_documents
    assert answer


def test_documents_are_non_empty(answer_and_documents):
    """Test that the documents are non-empty."""
    _, documents = answer_and_documents
    assert documents


def test_answer_is_string(answer_and_documents):
    """Test that the answer is a string."""
    answer, _ = answer_and_documents
    assert isinstance(answer, str)


def test_documents_are_list_of_documents(answer_and_documents):
    """Test that the documents are a list of Documents."""
    _, documents = answer_and_documents
    assert isinstance(documents, list)
    for document in documents:
        assert isinstance(document, Document)


def test_answer_is_correct(answer_and_documents):
    """Test that the answer is correct."""
    answer, _ = answer_and_documents
    assert "sort" in answer.lower()


def test_documents_are_correct(answer_and_documents):
    """Test that the documents are correct."""
    _, documents = answer_and_documents
    assert documents == [Document(id="2", text="Den sorte kat hedder Sutsko.")]


@pytest.mark.parametrize(
    argnames=["config"],
    argvalues=[
        (dict(),),
        (dict(document_store=dict(name="JsonlDocumentStore")),),
        (
            dict(
                document_store=dict(name="PostgresDocumentStore"),
                generator=dict(name="OpenAIGenerator"),
            ),
        ),
        (
            dict(
                retriever=dict(
                    name="EmbeddingRetriever",
                    embedder=dict(
                        name="E5Embedder",
                        embedder_model_id="intfloat/multilingual-e5-small",
                    ),
                    embedding_store=dict(name="NumpyEmbeddingStore"),
                )
            ),
        ),
        (
            dict(
                retriever=dict(
                    name="EmbeddingRetriever",
                    embedder=dict(name="E5Embedder"),
                    embedding_store=dict(name="NumpyEmbeddingStore"),
                ),
                generator=dict(
                    name="GGUFGenerator",
                    model_id="hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
                ),
            ),
        ),
    ],
    ids=[
        "default",
        "jsonl_document_store",
        "postgres_document_store_openai_generator",
        "embedding_retriever",
        "embedding_retriever_gguf_generator",
    ],
)
def test_from_config(config):
    """Test that a RAG system can be initialised from a config."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as config_file:
        config_file.write(json.dumps(config))
        config_file.seek(0)
        rag_system = RagSystem.from_config(config_file=config_file.name)

        # Check that the component has been initialised correctly
        for component_name, component_config in config.items():
            component_class_name = component_config["name"]
            component = getattr(rag_system, component_name)
            assert component.__class__.__name__ == component_class_name
