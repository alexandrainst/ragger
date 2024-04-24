"""Unit tests for the `demo` module."""

import json
import multiprocessing
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Generator

import pytest
from gradio_client import Client
from omegaconf import DictConfig
from ragger.demo import Demo


class TestDemo:
    """Tests for the `Demo` class."""

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
                    demo=dict(
                        host="localhost",
                        port=7860,
                        share=False,
                        password_protected=False,
                    ),
                )
            )

    def test_init(self, valid_config):
        """Test the initialisation of the demo."""
        demo = Demo(config=valid_config)
        assert demo.config == valid_config
        assert demo.rag_system is not None

    def test_launch(self, valid_config):
        """Test the launching of the demo."""
        # Launch the demo in a separate process
        demo = Demo(config=valid_config)
        demo_process = multiprocessing.Process(target=demo.launch)
        demo_process.start()

        # Wait for the demo to start
        sleep(5)
        client = Client(f"http://{valid_config.demo.host}:{valid_config.demo.port}")
        api = client.view_api(return_format="dict")
        assert api["named_endpoints"]["/RAG System"]
        demo_process.terminate()
