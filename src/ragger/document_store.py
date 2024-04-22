"""Store and fetch documents from a database."""

from abc import ABC, abstractmethod

from omegaconf import DictConfig


class DocumentStore(ABC):
    """An abstract document store, which fetches documents from a database."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the document store.

        Args:
            config:
                The Hydra configuration.
        """
        self.config = config

    @abstractmethod
    def __getitem__(self, index: str):
        """Fetch a document by its ID."""
        ...


class JsonlDocumentStore(DocumentStore):
    """A document store that fetches documents from a JSONL file."""

    pass
