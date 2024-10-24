"""Unit tests for the `demo` module."""

import sqlite3
import typing
from pathlib import Path
from tempfile import NamedTemporaryFile

import gradio as gr
import pytest

from ragger.demo import Demo


@pytest.fixture(scope="module")
def demo(rag_system) -> typing.Generator[Demo, None, None]:
    """A RAG demo for testing."""
    with NamedTemporaryFile(mode="w", suffix=".db") as file:
        demo = Demo(rag_system=rag_system, feedback_db_path=Path(file.name))
        yield demo
        demo.close()
        file.close()


def test_initialisation(demo):
    """Test the initialisation of the demo."""
    assert isinstance(demo, Demo)


def test_initialisation_with_feedback(rag_system):
    """Test the initialisation of the demo."""
    with NamedTemporaryFile(mode="w", suffix=".db") as file:
        demo = Demo(
            feedback_mode="feedback",
            rag_system=rag_system,
            feedback_db_path=Path(file.name),
        )
        with sqlite3.connect(database=demo.feedback_db_path) as connection:
            assert (
                connection.execute("""
                SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'
            """).fetchone()
                is not None
            )
        file.close()


def test_initialisation_with_strict_feedback(rag_system):
    """Test the initialisation of the demo."""
    with NamedTemporaryFile(mode="w", suffix=".db") as file:
        demo = Demo(
            feedback_mode="strict-feedback",
            rag_system=rag_system,
            feedback_db_path=Path(file.name),
        )
        with sqlite3.connect(database=demo.feedback_db_path) as connection:
            assert (
                connection.execute("""
                SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'
            """).fetchone()
                is not None
            )
        file.close()


def test_build(demo):
    """Test that the demo can be built."""
    blocks = demo.build_demo()
    assert isinstance(blocks, gr.Blocks)
