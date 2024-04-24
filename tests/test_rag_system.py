"""Unit tests for the `rag_system` module."""

import json
from tempfile import NamedTemporaryFile
from typing import Generator

import pytest
from omegaconf import DictConfig
from ragger.rag_system import RagSystem
from ragger.utils import Document, GeneratedAnswer


class TestRagSystem:
    """Tests for the `RagSystem` class."""

    @pytest.fixture(scope="class")
    def valid_config(self) -> Generator[DictConfig, None, None]:
        """A valid configuration for testing the RagSystem."""
        yield DictConfig(
            dict(
                document_store=dict(jsonl=dict(filename="data.jsonl")),
                embedder=dict(
                    e5=dict(
                        model_id="intfloat/multilingual-e5-small",
                        document_text_field="text",
                    )
                ),
                embedding_store=dict(numpy=dict(num_documents_to_retrieve=2)),
                generator=dict(
                    openai=dict(
                        api_key_variable_name="OPENAI_API_KEY",
                        model="gpt-3.5-turbo",
                        temperature=0.0,
                        stream=False,
                        timeout=60,
                        max_tokens=128,
                    )
                ),
            )
        )

    @pytest.fixture(scope="class")
    def invalid_config(self, valid_config) -> Generator[DictConfig, None, None]:
        """An invalid configuration for testing the RagSystem."""
        yield valid_config(
            embedding_store=dict(
                non_implemented_type=dict(num_documents_to_retrieve=2)
            ),
            generator=dict(
                non_implemented_type=dict(api_key_variable_name="OPENAI_API_KEY")
            ),
        )

    @pytest.fixture(scope="class")
    def valid_rag_system(self, valid_config) -> Generator[RagSystem, None, None]:
        """Initialise a RagSystem for testing."""
        with NamedTemporaryFile(mode="w", suffix=".jsonl") as file:
            # Create a JSONL file with some documents
            data_dicts = [
                dict(id="1", text="Den hvide og grå kat hedder Sjusk."),
                dict(id="2", text="Den sorte og hvide kat hedder Sutsko."),
            ]
            data_str = "\n".join(json.dumps(data_dict) for data_dict in data_dicts)
            file.write(data_str)
            file.flush()
            system = RagSystem(config=valid_config)
            yield system

    @pytest.fixture(scope="class")
    def invalid_rag_system(self, invalid_config) -> Generator[RagSystem, None, None]:
        """Initialise a RagSystem for testing."""
        with NamedTemporaryFile(mode="w", suffix=".jsonl") as file:
            # Create a JSONL file with some documents
            data_dicts = [
                dict(id="1", text="Den hvide og grå kat hedder Sjusk."),
                dict(id="2", text="Den sorte og hvide kat hedder Sutsko."),
            ]
            data_str = "\n".join(json.dumps(data_dict) for data_dict in data_dicts)
            file.write(data_str)
            file.flush()
            system = RagSystem(config=invalid_config)
            yield system

    def test_initialisation(self, valid_rag_system):
        """Test that the RagSystem can be initialised."""
        assert valid_rag_system

    def test_compile(self, valid_rag_system):
        """Test that the RagSystem can compile."""
        valid_rag_system.compile()
        assert valid_rag_system.document_store
        assert valid_rag_system.embedder
        assert valid_rag_system.embedding_store

    def test_answer(self, valid_rag_system):
        """Test that the RagSystem can answer a query."""
        answer, documents = valid_rag_system.answer(
            "Hvad hedder den hvide og sorte kat?"
        )
        assert answer
        assert documents
        assert isinstance(answer, str)
        assert isinstance(documents, list)
        for document in documents:
            assert isinstance(document, Document)
        assert len(documents) == 2
        expected_answer = GeneratedAnswer(
            answer="Den sorte og hvide kat hedder Sutsko.", sources=["2"]
        )
        assert answer == expected_answer.answer

    def test_error_if_invalid_config(self, invalid_rag_system):
        """Test that the RagSystem raises an error if the configuration is invalid."""
        with pytest.raises(ValueError):
            invalid_rag_system.compile()
        with pytest.raises(ValueError):
            invalid_rag_system.answer("Hvad hedder den hvide og sorte kat?")
