"""Unit tests for the `demo` module."""

from ragger.demo import Demo


class TestDemo:
    """Tests for the `Demo` class."""

    def test_initialisation(self, valid_config):
        """Test the initialisation of the demo."""
        demo = Demo(config=valid_config)
        assert demo.config == valid_config
        assert demo.rag_system is not None
