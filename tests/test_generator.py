"""Unit tests for the `generator` module."""

import typing

import pytest
import torch
from ragger.data_models import GeneratedAnswer
from ragger.generator import Generator, OpenaiGenerator, VllmGenerator


@pytest.fixture(scope="module")
def query() -> typing.Generator[str, None, None]:
    """A query for testing the generator."""
    yield "Hvad farve er Pjuskebusk?"


class TestOpenaiGenerator:
    """Tests for the `OpenaiGenerator` class."""

    def test_is_generator(self) -> None:
        """Test that the OpenaiGenerator is a Generator."""
        assert issubclass(OpenaiGenerator, Generator)

    def test_initialisation(self) -> None:
        """Test that the generator is initialised correctly."""
        assert OpenaiGenerator()

    def test_generate(self, query, documents) -> None:
        """Test that the generator generates an answer."""
        answer = OpenaiGenerator().generate(query=query, documents=documents)
        expected = GeneratedAnswer(answer="Pjuskebusk er rød.", sources=["3"])
        assert answer == expected

    def test_streaming(self, query, documents):
        """Test that the generator streams answers."""
        generator = OpenaiGenerator(stream=True)
        answer = generator.generate(query=query, documents=documents)
        assert isinstance(answer, typing.Generator)
        for partial_answer in answer:
            assert isinstance(partial_answer, GeneratedAnswer)

    def test_error_if_not_json(self, query, documents) -> None:
        """Test that the generator raises an error if the output is not JSON."""
        generator = OpenaiGenerator(max_output_tokens=3)
        answer = generator.generate(query=query, documents=documents)
        expected = GeneratedAnswer(answer="Not JSON-decodable.", sources=[])
        assert answer == expected

    def test_error_if_not_valid_types(self, query, documents) -> None:
        """Test that the generator raises an error if the JSON isn't valid."""
        generator = OpenaiGenerator()
        bad_prompt = 'Inkludér kilderne i key\'en "kilder" i stedet for "sources".'
        answer = generator.generate(query=f"{query}\n{bad_prompt}", documents=documents)
        expected = GeneratedAnswer(answer="JSON not valid.", sources=[])
        assert answer == expected


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="No GPU available.")
class TestVllmGenerator:
    """Tests for the `VllmGenerator` class."""

    @pytest.fixture(scope="class")
    def model_id(self) -> typing.Generator[str, None, None]:
        """The vLLM model ID for testing."""
        yield "ThatsGroes/munin-SkoleGPTOpenOrca-7b-16bit"

    def test_is_generator(self) -> None:
        """Test that the VllmGenerator is a Generator."""
        assert issubclass(VllmGenerator, Generator)

    def test_initialisation(self, model_id) -> None:
        """Test that the generator is initialised correctly."""
        assert VllmGenerator(model_id=model_id)

    def test_generate(self, model_id, query, documents) -> None:
        """Test that the generator generates an answer."""
        generator = VllmGenerator(model_id=model_id)
        answer = generator.generate(query=query, documents=documents)
        expected = GeneratedAnswer(answer="Pjuskebusk er rød.", sources=["3"])
        assert answer == expected

    def test_error_if_not_json(self, model_id, query, documents) -> None:
        """Test that the generator raises an error if the output is not JSON."""
        generator = VllmGenerator(model_id=model_id, max_output_tokens=1)
        with pytest.raises(ValueError):
            generator.generate(query=query, documents=documents)
