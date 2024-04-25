"""Unit tests for the `demo` module."""

from ragger.demo import Demo


class TestDemo:
    """Tests for the `Demo` class."""

    def test_initialisation(self, config):
        """Test the initialisation of the demo."""
        demo = Demo(config=config)
        assert demo.config == config
        assert demo.rag_system is not None
