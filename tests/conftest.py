"""Test fixtures used throughout the test suite."""

import json
import typing
from tempfile import NamedTemporaryFile

import pytest
from omegaconf import DictConfig


@pytest.fixture(scope="session")
def jsonl_filename() -> typing.Generator[str, None, None]:
    """A JSONL file with some example documents."""
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
        yield file.name


@pytest.fixture(scope="session")
def db_filename() -> typing.Generator[str, None, None]:
    """A SQLite database file."""
    with NamedTemporaryFile(mode="w", suffix=".db") as file:
        yield file.name


@pytest.fixture(scope="session")
def full_config(
    jsonl_filename, db_filename
) -> typing.Generator[DictConfig, None, None]:
    """A valid Hydra configuration."""
    yield DictConfig(
        dict(
            dirs=dict(
                data="data",
                raw="raw",
                processed="processed",
                final="final",
                models="models",
            ),
            document_store=dict(name="jsonl", filename=jsonl_filename),
            embedder=dict(
                name="e5",
                model_id="intfloat/multilingual-e5-small",
                document_text_field="text",
            ),
            embedding_store=dict(
                name="numpy", num_documents_to_retrieve=3, embedding_path=""
            ),
            generator=dict(
                name="openai",
                api_key_variable_name="OPENAI_API_KEY",
                model="gpt-3.5-turbo",
                temperature=0.0,
                stream=False,
                timeout=60,
                max_tokens=128,
            ),
            demo=dict(
                name="danish",
                host="localhost",
                port=7860,
                share=False,
                password_protected=False,
                mode="no_feedback",
                db_path=db_filename,
            ),
        )
    )
