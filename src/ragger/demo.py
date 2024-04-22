"""A Gradio demo of the RAG system."""

from omegaconf import DictConfig

from .rag_system import RagSystem


class Demo:
    """A Gradio demo of the RAG system."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the demo.

        Args:
            config:
                The Hydra configuration.
        """
        self.config = config
        self.rag_system = RagSystem(config=config)

    def launch(self) -> None:
        """Launch the demo."""
        raise NotImplementedError
