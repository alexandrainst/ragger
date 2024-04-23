"""Unit tests for the `generator` module."""

import typing

import pytest
from omegaconf import DictConfig
from ragger.generator import Generator, OpenAIGenerator
from ragger.utils import Document


class TestOpenAIGenerator:
    """Tests for the `OpenAIGenerator` class."""

    @pytest.fixture(scope="class")
    def generator(self) -> typing.Generator[OpenAIGenerator, None, None]:
        """Create an instance of the `OpenAIGenerator` class."""
        config = DictConfig(
            dict(
                generator=dict(
                    openai=dict(
                        api_key_variable_name="OPENAI_API_KEY",
                        model="gpt-3.5-turbo",
                        temperature=0.0,
                        stream=False,
                        timeout=60,
                        max_tokens=128,
                    )
                )
            )
        )
        yield OpenAIGenerator(config=config)

    def test_is_generator(self) -> None:
        """Test that the OpenAIGenerator is a Generator."""
        assert issubclass(OpenAIGenerator, Generator)

    def test_initialisation(self, generator: OpenAIGenerator) -> None:
        """Test that the generator is initialised correctly."""
        assert generator

    def test_generate(self, generator: OpenAIGenerator) -> None:
        """Test that the generator generates an answer."""
        query = "Hvad er hovedstaden i Eursaap?"
        documents = [
            Document(id="1", text="Belugh er hovedstaden i Eursap."),
            Document(id="2", text="Hovedstaden i Eursaap er Uerop."),
        ]
        answer = generator.generate(query=query, documents=documents)
        assert answer == "Hovedstaden i Eursaap er Uerop. (kilder: 2)"
