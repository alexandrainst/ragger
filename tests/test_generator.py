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
        """A configuration used for testing the OpenAIGenerator."""
        yield DictConfig(
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

    @pytest.fixture(scope="class")
    def config_with_few_max_tokens(self) -> typing.Generator[DictConfig, None, None]:
        """A configuration where the maximum number of tokens is very low."""
        yield DictConfig(
            dict(
                generator=dict(
                    openai=dict(
                        api_key_variable_name="OPENAI_API_KEY",
                        model="gpt-3.5-turbo",
                        temperature=0.0,
                        stream=False,
                        timeout=60,
                        max_tokens=1,
                    )
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

    def test_initialisation(self, config: DictConfig) -> None:
        """Test that the generator is initialised correctly."""
        assert OpenAIGenerator(config=config)

    def test_generate(
        self, config: DictConfig, query: str, documents: list[Document]
    ) -> None:
        """Test that the generator generates an answer."""
        generator = OpenAIGenerator(config=config)
        answer = generator.generate(query=query, documents=documents)
        expected = GeneratedAnswer(
            answer="Hovedstaden i Eursaap er Uerop.", sources=["2"]
        )
        assert answer == expected

    def test_error_if_not_json(
        self,
        config_with_few_max_tokens: DictConfig,
        query: str,
        documents: list[Document],
    ) -> None:
        """Test that the generator raises an error if the output is not JSON."""
        generator = OpenAIGenerator(config=config_with_few_max_tokens)
        with pytest.raises(ValueError):
            generator.generate(query=query, documents=documents)

    def test_error_if_not_valid_types(
        self, config: DictConfig, query: str, documents: list[Document]
    ) -> None:
        """Test that the generator raises an error if the output is not JSON."""
        generator = OpenAIGenerator(config=config)
        bad_prefix = "I dit svar skal du ikke inkludere kilderne, kun 'answer' key'en."
        with pytest.raises(ValueError):
            generator.generate(query=f"{bad_prefix}\n{query}", documents=documents)
