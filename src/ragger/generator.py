"""Generation of an answer from a query and a list of relevant documents."""

from abc import ABC, abstractmethod

from omegaconf import DictConfig

from .utils import Document


class Generator(ABC):
    """An abstract generator of answers from a query and relevant documents."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the generator.

        Args:
            config:
                The Hydra configuration.
        """
        self.config = config

    @abstractmethod
    def generate(self, query: str, documents: list[Document]) -> str:
        """Generate an answer from a query and relevant documents.

        Args:
            query:
                The query to answer.
            documents:
                The relevant documents.

        Returns:
            The generated answer.
        """
        ...


class OpenAIGenerator(Generator):
    """A generator that uses an OpenAI model to generate answers."""

    pass
