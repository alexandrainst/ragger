"""Test fixtures used throughout the test suite."""

import json
from tempfile import NamedTemporaryFile
from typing import Generator

import pytest
from omegaconf import DictConfig


@pytest.fixture(scope="session")
def valid_config() -> Generator[DictConfig, None, None]:
    """A valid Hydra configuration."""
    with NamedTemporaryFile(mode="w", suffix=".jsonl") as file:
        data_dicts = [
            dict(id="1", text="Den hvide kat hedder Sjusk."),
            dict(id="2", text="Den sorte kat hedder Sutsko."),
            dict(id="3", text="Den røde kat hedder Pjuskebusk."),
            dict(id="4", text="Den grønne kat hedder Sjask."),
            dict(id="5", text="Den blå kat hedder Sky."),
        ]
        data_str = "\n".join(json.dumps(data_dict) for data_dict in data_dicts)
        file.write(data_str)
        file.flush()

        yield DictConfig(
            dict(
                dirs=dict(data="data", raw="raw", processed="processed", final="final"),
                document_store=dict(type="jsonl", jsonl=dict(filename=file.name)),
                embedder=dict(
                    type="e5",
                    e5=dict(
                        model_id="intfloat/multilingual-e5-small",
                        document_text_field="text",
                    ),
                ),
                embedding_store=dict(
                    type="numpy", numpy=dict(num_documents_to_retrieve=3)
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
                    host="localhost", port=7860, share=False, password_protected=False
                ),
            )
        )
