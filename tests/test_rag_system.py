"""Unit tests for the `rag_system` module."""

import json
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
        valid_config.document_store.type = "invalid-type"
        valid_config.generator.type = "invalid-type"
        yield valid_config

    @pytest.fixture(scope="class")
    def valid_rag_system(self, valid_config) -> Generator[RagSystem, None, None]:
        """Initialise a RagSystem for testing."""
        system = RagSystem(config=valid_config)
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
        answer, documents = valid_rag_system.answer("Hvad farve har Sutsko?")
        assert answer
        assert documents
        assert isinstance(answer, str)
        assert isinstance(documents, list)
        for document in documents:
            assert isinstance(document, Document)
        expected_answer = (
            "Sort",
            [Document(id="2", text="Den sorte kat hedder Sutsko.")],
        )
        assert answer == expected_answer[0]
        assert documents == expected_answer[1]

    def test_error_if_invalid_config(self, invalid_config):
        """Test that the RagSystem raises an error if the configuration is invalid."""
        with pytest.raises(ValueError):
            RagSystem(config=invalid_config)
