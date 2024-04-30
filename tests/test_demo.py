"""Unit tests for the `demo` module."""

import sqlite3
from pathlib import Path

from ragger.demo import Demo


class TestDemo:
    """Tests for the `Demo` class."""

    def test_initialisation_no_feedback(self, config):
        """Test the initialisation of the demo."""
        demo = Demo(config=config)
        assert demo.config == config
        assert demo.rag_system is not None

    def test_initialisation_feedback(self, config):
        """Test the initialisation of the demo."""
        for mode in ["strict_feedback", "feedback"]:
            config.demo.mode = mode
            demo = Demo(config=config)
            assert demo.config == config
            assert demo.rag_system is not None
            db_path = Path(config.dirs.data) / config.demo.db_path
            connection = sqlite3.connect(db_path)
            assert connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
            ).fetchone()
            connection.close()
