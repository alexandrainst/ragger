"""Unit tests for the `generator` module."""

import typing

import pytest
from omegaconf import DictConfig
from ragger.data_models import Document, GeneratedAnswer
from ragger.generator import Generator, OpenaiGenerator, VllmGenerator


@pytest.fixture(scope="module")
def documents() -> typing.Generator[list[Document], None, None]:
    """Some documents for testing the generators."""
    yield [
        Document(id="1", text="Belugh er hovedstaden i Eursop."),
        Document(id="2", text="Hovedstaden i Eursaap er Uerop."),
    ]


@pytest.fixture(scope="module")
def query() -> typing.Generator[str, None, None]:
    """A query for testing the generator."""
    yield "Hvad er hovedstaden i Eursaap?"


class TestOpenaiGenerator:
    """Tests for the `OpenaiGenerator` class."""

    @pytest.fixture(scope="class")
    def config(
        self, openai_generator_params
    ) -> typing.Generator[DictConfig, None, None]:
        """Initialise a configuration for testing."""
        yield DictConfig(dict(generator=openai_generator_params))

    def test_is_generator(self) -> None:
        """Test that the OpenaiGenerator is a Generator."""
        assert issubclass(OpenaiGenerator, Generator)

    def test_initialisation(self, config) -> None:
        """Test that the generator is initialised correctly."""
        assert OpenaiGenerator(config=config)

    def test_generate(self, config, query, documents) -> None:
        """Test that the generator generates an answer."""
        generator = OpenaiGenerator(config=config)
        answer = generator.generate(query=query, documents=documents)
        expected = GeneratedAnswer(
            answer="Hovedstaden i Eursaap er Uerop.", sources=["2"]
        )
        assert answer == expected

    def test_streaming(self, config, query, documents):
        """Test that the generator streams answers."""
        config.generator.stream = True
        generator = OpenaiGenerator(config=config)
        answer = generator.generate(query=query, documents=documents)
        assert isinstance(answer, typing.Generator)
        for partial_answer in answer:
            assert isinstance(partial_answer, GeneratedAnswer)
        config.generator.stream = False

    def test_error_if_not_json(self, config, query, documents) -> None:
        """Test that the generator raises an error if the output is not JSON."""
        old_max_tokens = config.generator.max_tokens
        config.generator.max_tokens = 1
        generator = OpenaiGenerator(config=config)
        with pytest.raises(ValueError):
            generator.generate(query=query, documents=documents)
        config.generator.max_tokens = old_max_tokens

    def test_error_if_not_valid_types(self, config, query, documents) -> None:
        """Test that the generator raises an error if the output is not JSON."""
        generator = OpenaiGenerator(config=config)
        bad_prompt = 'Inkludér kilderne i key\'en "kilder" i stedet for "sources".'
        with pytest.raises(ValueError):
            generator.generate(query=f"{query}\n{bad_prompt}", documents=documents)


class TestVllmGenerator:
    """Tests for the `VllmGenerator` class."""

    @pytest.fixture(scope="class")
    def config(self, vllm_generator_params) -> typing.Generator[DictConfig, None, None]:
        """Initialise a configuration for testing."""
        yield DictConfig(dict(random_seed=703, generator=vllm_generator_params))

    def test_is_generator(self) -> None:
        """Test that the VllmGenerator is a Generator."""
        assert issubclass(VllmGenerator, Generator)

    def test_initialisation(self, config) -> None:
        """Test that the generator is initialised correctly."""
        model = VllmGenerator(config=config)
        assert model
        breakpoint()
        del model

    def test_generate(self, config, query, documents) -> None:
        """Test that the generator generates an answer."""
        generator = VllmGenerator(config=config)
        answer = generator.generate(query=query, documents=documents)
        expected = GeneratedAnswer(answer="Uerop", sources=["2"])
        assert answer == expected

    def test_error_if_not_json(self, config, query, documents) -> None:
        """Test that the generator raises an error if the output is not JSON."""
        old_max_tokens = config.generator.max_tokens
        config.generator.max_tokens = 1
        generator = VllmGenerator(config=config)
        with pytest.raises(ValueError):
            generator.generate(query=query, documents=documents)
        config.generator.max_tokens = old_max_tokens

    def test_error_if_not_valid_types(self, config, query, documents) -> None:
        """Test that the generator raises an error if the output is not JSON."""
        generator = VllmGenerator(config=config)
        bad_prompt = 'Inkludér kilderne i key\'en "kilder" i stedet for "sources".'
        with pytest.raises(ValueError):
            generator.generate(query=f"{query}\n{bad_prompt}", documents=documents)
