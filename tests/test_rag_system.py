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
    def rag_system(self) -> Generator[RagSystem, None, None]:
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
            config = DictConfig(
                dict(
                    document_store=dict(jsonl=dict(filename=file.name)),
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
            system = RagSystem(config=config)
            yield system

    def test_initialisation(self, rag_system):
        """Test that the RagSystem can be initialised."""
        assert rag_system

    def test_compile(self, rag_system):
        """Test that the RagSystem can compile."""
        rag_system.compile()
        assert rag_system.document_store
        assert rag_system.embedder
        assert rag_system.embedding_store

    def test_answer(self, rag_system):
        """Test that the RagSystem can answer a query."""
        answer, documents = rag_system.answer("Hvad hedder den hvide og sorte kat?")
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
