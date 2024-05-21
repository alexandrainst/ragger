"""Data models used in the RAG system."""

import typing
from abc import ABC, abstractmethod
from typing import Annotated, Type

import annotated_types
import numpy as np
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict

Index = str


class Document(BaseModel):
    """A document to be stored in a document store."""

    id: Index
    text: str


class Embedding(BaseModel):
    """An embedding of a document."""

    id: Index
    embedding: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


class GeneratedAnswer(BaseModel):
    """A generated answer to a question."""

    sources: list[Annotated[Index, annotated_types.Len(min_length=1)]]
    answer: str = ""


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
    def __getitem__(self, index: Index) -> Document:
        """Fetch a document by its ID.

        Args:
            index:
                The ID of the document to fetch.

        Returns:
            The document with the given ID.
        """
        ...

    @abstractmethod
    def get_all_documents(self) -> list[Document]:
        """Fetch all documents from the store.

        Returns:
            A list of all documents in the store.
        """
        ...


class Embedder(ABC):
    """An abstract embedder, which embeds documents using a pre-trained model."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the embedder.

        Args:
            config:
                The Hydra configuration.
        """
        self.config = config

    def compile(self) -> None:
        """Compile the embedder.

        This method loads any necessary resources and prepares the embedder for use.
        """
        pass

    @abstractmethod
    def embed_documents(self, documents: list[Document]) -> list[Embedding]:
        """Embed a list of documents.

        Args:
            documents:
                A list of documents to embed.

        Returns:
            An array of embeddings, where each row corresponds to a document.
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query.

        Args:
            query:
                A query.

        Returns:
            The embedding of the query.
        """
        ...

    @abstractmethod
    def tokenize(self, text: str | list[str]) -> np.ndarray:
        """Tokenize a text.

        Args:
            text:
                The text or texts to tokenize.

        Returns:
            The tokens of the text.
        """
        ...

    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """The maximum length of the context that the embedder can handle."""
        ...


class EmbeddingStore(ABC):
    """An abstract embedding store, which fetches embeddings from a database."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the embedding store.

        Args:
            config:
                The Hydra configuration.
        """
        self.config = config

    @abstractmethod
    def add_embeddings(self, embeddings: list[Embedding]) -> None:
        """Add embeddings to the store.

        Args:
            embeddings:
                A list of embeddings to add to the store.
        """
        ...

    @abstractmethod
    def get_nearest_neighbours(self, embedding: np.ndarray) -> list[Index]:
        """Get the nearest neighbours to a given embedding.

        Args:
            embedding:
                The embedding to find nearest neighbours for.

        Returns:
            A list of indices of the nearest neighbours.
        """
        ...

    @abstractmethod
    def document_exists_in_store(self, document_id: Index) -> bool:
        """Check if a document exists in the store.

        Args:
            document_id:
                The ID of the document to check.

        Returns:
            Whether the document exists in the store.
        """
        ...


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
    def generate(
        self, query: str, documents: list[Document]
    ) -> GeneratedAnswer | typing.Generator[GeneratedAnswer, None, None]:
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


class Components(BaseModel):
    """The components of the RAG system."""

    document_store: Type[DocumentStore] | None = None
    embedder: Type[Embedder] | None = None
    embedding_store: Type[EmbeddingStore] | None = None
    generator: Type[Generator] | None = None
