"""Unit tests for the `demo` module."""

import sqlite3
from copy import deepcopy
from pathlib import Path

from ragger.demo import Demo


def test_initialisation_no_feedback(full_config):
    """Test the initialisation of the demo."""
    demo = Demo(config=full_config)
    assert demo.config == full_config
    assert demo.rag_system is not None


def test_initialisation_feedback(full_config):
    """Test the initialisation of the demo."""
    config_copy = deepcopy(full_config)
    for mode in ["strict-feedback", "feedback"]:
        config_copy.demo.mode = mode
        Demo(config=config_copy)
        with sqlite3.connect(
            database=Path(config_copy.dirs.data) / config_copy.demo.db_path
        ) as connection:
            assert connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
            ).fetchone()
