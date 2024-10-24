"""Unit tests for the `rag_system` module."""

import typing

import pytest

from ragger.data_models import Document


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
