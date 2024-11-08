"""Tests for the `retriever` module."""

import typing

import pytest

import alexandrainst_ragger.retriever
from alexandrainst_ragger.data_models import Retriever


@pytest.fixture(
    scope="module",
    params=[
        cls
        for cls in vars(alexandrainst_ragger.retriever).values()
        if isinstance(cls, type) and issubclass(cls, Retriever) and cls is not Retriever
    ],
)
def retriever(
    request, special_kwargs, rag_system
) -> typing.Generator[Retriever, None, None]:
    """Initialise a retriever for testing."""
    retriever_cls = request.param
    retriever: Retriever = request.param(
        **special_kwargs.get(retriever_cls.__name__, {})
    )
    retriever.compile(
        document_store=rag_system.document_store, generator=rag_system.generator
    )
    yield retriever


def test_initialisation(retriever):
    """Test that the retriever can be initialised."""
    assert isinstance(retriever, Retriever)


def test_retrieve(retriever, query, documents):
    """Test that the retriever can retrieve text."""
    document_ids = retriever.retrieve(query=query, num_docs=5)
    assert document_ids[0] == documents[0].id
