"""Unit tests for the `rag_system` module."""

import json
from copy import deepcopy
from tempfile import NamedTemporaryFile
from typing import Generator

import pytest
from omegaconf import DictConfig
from ragger.rag_system import RagSystem
from ragger.utils import Document


class TestRagSystem:
    """Tests for the `RagSystem` class."""

    @pytest.fixture(scope="class")
    def valid_config(self) -> Generator[DictConfig, None, None]:
        """A valid configuration for testing the RagSystem."""
        with NamedTemporaryFile(mode="w", suffix=".jsonl") as file:
            # Create a JSONL file with some documents
            data_dicts = [
                dict(id="1", text="Den hvide kat hedder Sjusk."),
                dict(id="2", text="Den sorte kat hedder Sutsko."),
            ]
            data_str = "\n".join(json.dumps(data_dict) for data_dict in data_dicts)
            file.write(data_str)
            file.flush()

            yield DictConfig(
                dict(
                    document_store=dict(type="jsonl", jsonl=dict(filename=file.name)),
                    embedder=dict(
                        type="e5",
                        e5=dict(
                            model_id="intfloat/multilingual-e5-small",
                            document_text_field="text",
                        ),
                    ),
                    embedding_store=dict(
                        type="numpy", numpy=dict(num_documents_to_retrieve=2)
                    ),
                    generator=dict(
                        type="openai",
                        openai=dict(
                            api_key_variable_name="OPENAI_API_KEY",
                            model="gpt-3.5-turbo",
                            temperature=0.0,
                            stream=False,
                            timeout=60,
                            max_tokens=128,
                        ),
                    ),
                )
            )

    @pytest.fixture(scope="class")
    def invalid_config(self, valid_config) -> Generator[DictConfig, None, None]:
        """An invalid configuration for testing the RagSystem."""
        config = deepcopy(valid_config)
        config.document_store.type = "invalid-type"
        config.generator.type = "invalid-type"
        yield config

    @pytest.fixture(scope="class")
    def valid_rag_system(self, valid_config) -> Generator[RagSystem, None, None]:
        """Initialise a RagSystem for testing."""
        yield RagSystem(config=valid_config)

    @pytest.fixture(scope="class")
    def compiled_rag_system(self, valid_config) -> Generator[RagSystem, None, None]:
        """A compiled RagSystem for testing."""
        rag_system = RagSystem(config=valid_config)
        rag_system.compile()
        yield rag_system

    @pytest.fixture(scope="class")
    def answer_and_documents(self, compiled_rag_system):
        """An answer and supporting documents for testing."""
        yield compiled_rag_system.answer("Hvad farve har Sutsko?")

    def test_initialisation(self, valid_rag_system):
        """Test that the RagSystem can be initialised."""
        assert valid_rag_system

    def test_compile(self, compiled_rag_system):
        """Test that the RagSystem can be compiled."""
        assert compiled_rag_system.document_store
        assert compiled_rag_system.embedder
        assert compiled_rag_system.embedding_store

    def test_answer_is_non_empty(self, answer_and_documents):
        """Test that the answer is non-empty."""
        answer, _ = answer_and_documents
        assert answer

    def test_documents_are_non_empty(self, answer_and_documents):
        """Test that the documents are non-empty."""
        _, documents = answer_and_documents
        assert documents

    def test_answer_is_string(self, answer_and_documents):
        """Test that the answer is a string."""
        answer, _ = answer_and_documents
        assert isinstance(answer, str)

    def test_documents_are_list_of_documents(self, answer_and_documents):
        """Test that the documents are a list of Documents."""
        _, documents = answer_and_documents
        assert isinstance(documents, list)
        for document in documents:
            assert isinstance(document, Document)

    def test_answer_is_correct(self, answer_and_documents):
        """Test that the answer is correct."""
        answer, _ = answer_and_documents
        assert "sort" in answer.lower()

    def test_documents_are_correct(self, answer_and_documents):
        """Test that the documents are correct."""
        _, documents = answer_and_documents
        assert documents == [Document(id="2", text="Den sorte kat hedder Sutsko.")]

    def test_error_if_invalid_config(self, invalid_config):
        """Test that the RagSystem raises an error if the configuration is invalid."""
        with pytest.raises(ValueError):
            RagSystem(config=invalid_config)
