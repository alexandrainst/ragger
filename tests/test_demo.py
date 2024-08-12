"""Unit tests for the `demo` module."""

import sqlite3
import typing
from pathlib import Path
from tempfile import NamedTemporaryFile

from ragger.demo import Demo


def test_initialisation_no_feedback(rag_system):
    """Test the initialisation of the demo."""
    with NamedTemporaryFile(mode="w", suffix=".db") as file:
        Demo(rag_system=rag_system, feedback_db_path=Path(file.name))
        file.close()


def test_initialisation_feedback(rag_system):
    """Test the initialisation of the demo."""
    feedback_modes: list[typing.Literal["strict-feedback", "feedback"]] = [
        "strict-feedback",
        "feedback",
    ]
    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
    for feedback_mode in feedback_modes:
        with NamedTemporaryFile(mode="w", suffix=".db") as file:
            demo = Demo(
                feedback_mode=feedback_mode,
                rag_system=rag_system,
                feedback_db_path=Path(file.name),
            )
            with sqlite3.connect(database=demo.feedback_db_path) as connection:
                assert connection.execute(sql).fetchone()
            file.close()
