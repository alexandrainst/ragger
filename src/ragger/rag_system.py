"""The main entry point for the RAG system, orchestrating the other components."""

from omegaconf import DictConfig

from .document_store import DocumentStore
from .embedder import Embedder
from .embedding_store import EmbeddingStore
from .generator import Generator
from .utils import Document


class RagSystem:
    """The main entry point for the RAG system, orchestrating the other components."""

    def __init__(self, config: DictConfig):
        """Initialise the RAG system.

        Args:
            config:
                The Hydra configuration.
        """
        self.config = config
        self.document_store: DocumentStore
        self.embedder: Embedder
        self.embedding_store: EmbeddingStore
        self.generator: Generator
        self.compile()

    def compile(self) -> None:
        """Compile the RAG system.

        This builds the underlying embedding store and can be called whenever the data
        needs to be updated.
        """
        raise NotImplementedError

    def answer(self, query: str) -> tuple[str, list[Document]]:
        """Answer a query.

        Args:
            query:
                The query to answer.

        Returns:
            A tuple of the answer and the supporting documents.
        """
        raise NotImplementedError
