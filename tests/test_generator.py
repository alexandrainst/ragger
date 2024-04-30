"""Unit tests for the `generator` module."""

import typing

import pytest
from omegaconf import DictConfig
from ragger.generator import Generator, OpenAIGenerator
from ragger.utils import Document, GeneratedAnswer


class TestOpenAIGenerator:
    """Tests for the `OpenAIGenerator` class."""

    @pytest.fixture(scope="class")
    def config(self) -> typing.Generator[DictConfig, None, None]:
        """Initialise a configuration for testing."""
        yield DictConfig(
            dict(
                generator=dict(
                    name="openai",
                    api_key_variable_name="OPENAI_API_KEY",
                    model="gpt-3.5-turbo",
                    temperature=0.0,
                    stream=False,
                    timeout=60,
                    max_tokens=128,
                )
            )
        )

    @pytest.fixture(scope="class")
    def documents(self) -> typing.Generator[list[Document], None, None]:
        """Some documents for testing the OpenAIGenerator."""
        yield [
            Document(id="1", text="Belugh er hovedstaden i Eursap."),
            Document(id="2", text="Hovedstaden i Eursaap er Uerop."),
        ]

    @pytest.fixture(scope="class")
    def query(self) -> typing.Generator[str, None, None]:
        """A query for testing the OpenAIGenerator."""
        yield "Hvad er hovedstaden i Eursaap?"

    def test_is_generator(self) -> None:
        """Test that the OpenAIGenerator is a Generator."""
        assert issubclass(OpenAIGenerator, Generator)

    def test_initialisation(self, config) -> None:
        """Test that the generator is initialised correctly."""
        assert OpenAIGenerator(config=config)

    def test_generate(self, config, query, documents) -> None:
        """Test that the generator generates an answer."""
        generator = OpenAIGenerator(config=config)
        answer = generator.generate(query=query, documents=documents)
        expected = GeneratedAnswer(
            answer="Hovedstaden i Eursaap er Uerop.", sources=["2"]
        )
        assert answer == expected

    def test_streaming(self, config, query, documents):
        """Test that the generator streams answers."""
        config.generator.stream = True
        generator = OpenAIGenerator(config=config)
        answer = generator.generate(query=query, documents=documents)
        assert isinstance(answer, typing.Generator)
        for partial_answer in answer:
            assert isinstance(partial_answer, GeneratedAnswer)
        config.generator.stream = False

    def test_error_if_not_json(self, config, query, documents) -> None:
        """Test that the generator raises an error if the output is not JSON."""
        old_max_tokens = config.generator.max_tokens
        config.generator.max_tokens = 1
        generator = OpenAIGenerator(config=config)
        with pytest.raises(ValueError):
            generator.generate(query=query, documents=documents)
        config.generator.max_tokens = old_max_tokens

    def test_error_if_not_valid_types(self, config, query, documents) -> None:
        """Test that the generator raises an error if the output is not JSON."""
        generator = OpenAIGenerator(config=config)
        bad_prefix = 'Inkludér svaret i key\'en "andet" i stedet for "answer".'
        with pytest.raises(ValueError):
            generator.generate(query=f"{bad_prefix}\n{query}", documents=documents)
