"""Unit tests for the `rag_system` module."""

import typing
from copy import deepcopy

import pytest
from ragger.rag_system import RagSystem
from ragger.utils import Document


@pytest.fixture(scope="module")
def compiled_rag_system(full_config) -> typing.Generator[RagSystem, None, None]:
    """A compiled RagSystem for testing."""
    rag_system = RagSystem(config=full_config)
    rag_system.compile()
    yield rag_system


@pytest.fixture(scope="module")
def answer_and_documents(compiled_rag_system):
    """An answer and supporting documents for testing."""
    yield compiled_rag_system.answer("Hvad farve har Sutsko?")


def test_initialisation(full_config):
    """Test that the RagSystem can be initialised."""
    assert RagSystem(config=full_config)


def test_compile(compiled_rag_system):
    """Test that the RagSystem can be compiled."""
    assert compiled_rag_system.document_store
    assert compiled_rag_system.embedder
    assert compiled_rag_system.embedding_store


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


def test_error_if_invalid_config(full_config):
    """Test that the RagSystem raises an error if the configuration is invalid."""
    invalid_config = deepcopy(full_config)
    invalid_config.document_store.name = "invalid-type"
    invalid_config.generator.name = "invalid-type"
    with pytest.raises(ValueError):
        RagSystem(config=invalid_config)
