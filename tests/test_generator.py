"""Unit tests for the `generator` module."""

import inspect
import typing

import pytest
from numpy.typing import NDArray

import ragger.generator
from ragger.data_models import GeneratedAnswer, Generator
from ragger.exceptions import MissingExtra, MissingPackage


@pytest.fixture(
    scope="module",
    params=[
        cls
        for cls in vars(ragger.generator).values()
        if inspect.isclass(object=cls)
        and cls is not NDArray  # numpy.typing.NDArray is not a class in Python 3.10
        and issubclass(cls, Generator)
        and cls is not Generator
    ],
)
def generator(
    request, special_kwargs, rag_system
) -> typing.Generator[Generator, None, None]:
    """Initialise a generator class for testing."""
    try:
        generator_cls = request.param
        generator: Generator = generator_cls(
            **special_kwargs.get(generator_cls.__name__, {})
        )
        generator.compile(
            document_store=rag_system.document_store, retriever=rag_system.retriever
        )
        yield generator
    except (MissingPackage, MissingExtra):
        pytest.skip("The generator could not be imported.")


def test_initialisation(generator) -> None:
    """Test that the generator is initialised correctly."""
    assert isinstance(generator, Generator)


def test_generate(query, documents, generator) -> None:
    """Test that the generator generates an answer."""
    answer = generator.generate(query=query, documents=documents)
    assert "Sjusk" in answer.answer


def test_streaming(generator, query, documents):
    """Test that the generator streams answers."""
    generator.stream = True
    answer = generator.generate(query=query, documents=documents)
    assert isinstance(answer, typing.Generator)
    for partial_answer in answer:
        assert isinstance(partial_answer, GeneratedAnswer)
    generator.stream = False


def test_error_if_not_json(generator, query, documents) -> None:
    """Test that the generator raises an error if the output is not JSON."""
    old_max_output_tokens = generator.max_output_tokens
    generator.max_output_tokens = 3
    answer = generator.generate(query=query, documents=documents)
    expected = GeneratedAnswer(answer="Not JSON-decodable.", sources=[])
    assert answer == expected
    generator.max_output_tokens = old_max_output_tokens
