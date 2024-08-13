"""Data models used in the RAG system."""

import typing
from abc import ABC, abstractmethod
from pathlib import Path

import annotated_types
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

Index = str


class Document(BaseModel):
    """A document to be stored in a document store."""

    id: Index
    text: str

    def __eq__(self, other: object) -> bool:
        """Check if two documents are equal.

        Args:
            other:
                The object to compare to.

        Returns:
            Whether the two documents are equal.
        """
        if not isinstance(other, Document):
            return False
        return self.id == other.id and self.text == other.text


class Embedding(BaseModel):
    """An embedding of a document."""

    id: Index
    embedding: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __eq__(self, other: object) -> bool:
        """Check if two embeddings are equal.

        Args:
            other:
                The object to compare to.

        Returns:
            Whether the two embeddings are equal.
        """
        if not isinstance(other, Embedding):
            return False
        return self.id == other.id and np.array_equal(self.embedding, other.embedding)


class GeneratedAnswer(BaseModel):
    """A generated answer to a question."""

    sources: typing.Annotated[
        list[typing.Annotated[Index, annotated_types.Len(min_length=1)]],
        Field(max_length=5),
    ]
    answer: str = ""


class DocumentStore(ABC):
    """An abstract document store, which fetches documents from a database."""

    path: Path

    def compile(
        self,
        embedder: "Embedder",
        embedding_store: "EmbeddingStore",
        generator: "Generator",
    ) -> None:
        """Compile the embedder.

        Args:
            embedder:
                The embedder to use.
            embedding_store:
                The embedding store to use.
            generator:
                The generator to use.
        """
        pass

    @abstractmethod
    def add_documents(self, documents: typing.Iterable[Document]) -> "DocumentStore":
        """Add documents to the store.

        Args:
            documents:
                An iterable of documents to add to the store.
        """
        ...

    @abstractmethod
    def remove(self) -> None:
        """Remove the document store."""
        ...

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
    def __contains__(self, index: Index) -> bool:
        """Check if a document with the given ID exists in the store.

        Args:
            index:
                The ID of the document to check.

        Returns:
            Whether the document exists in the store.
        """
        ...

    @abstractmethod
    def __iter__(self) -> typing.Generator[Document, None, None]:
        """Iterate over the documents in the store.

        Yields:
            The documents in the store.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of documents in the store.

        Returns:
            The number of documents in the store.
        """
        ...

    def __repr__(self) -> str:
        """Return a string representation of the document store.

        Returns:
            A string representation of the document store.
        """
        return f"{self.__class__.__name__}({len(self):,} documents)"


class Embedder(ABC):
    """An abstract embedder, which embeds documents using a pre-trained model."""

    embedding_dim: int

    def compile(
        self,
        document_store: "DocumentStore",
        embedding_store: "EmbeddingStore",
        generator: "Generator",
    ) -> None:
        """Compile the embedder.

        Args:
            document_store:
                The document store to use.
            embedding_store:
                The embedding store to use.
            generator:
                The generator to use.
        """
        pass

    @abstractmethod
    def embed_documents(self, documents: typing.Iterable[Document]) -> list[Embedding]:
        """Embed a list of documents.

        Args:
            documents:
                An iterable of documents to embed.

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

    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """The maximum length of the context that the embedder can handle."""
        ...

    def __repr__(self) -> str:
        """Return a string representation of the embedder.

        Returns:
            A string representation of the embedder.
        """
        return f"{self.__class__.__name__}({self.embedding_dim}-dim embeddings)"


class EmbeddingStore(ABC):
    """An abstract embedding store, which fetches embeddings from a database."""

    path: Path

    def compile(
        self,
        document_store: "DocumentStore",
        embedder: "Embedder",
        generator: "Generator",
    ) -> None:
        """Compile the embedder.

        Args:
            document_store:
                The document store to use.
            embedder:
                The embedder to use.
            generator:
                The generator to use.
        """
        pass

    @abstractmethod
    def add_embeddings(
        self, embeddings: typing.Iterable[Embedding]
    ) -> "EmbeddingStore":
        """Add embeddings to the store.

        Args:
            embeddings:
                An iterable of embeddings to add to the store.
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
    def clear(self) -> None:
        """Clear all embeddings from the store."""
        ...

    @abstractmethod
    def remove(self) -> None:
        """Remove the embedding store."""
        ...

    @abstractmethod
    def __getitem__(self, document_id: Index) -> Embedding:
        """Fetch an embedding by its document ID.

        Args:
            document_id:
                The ID of the document to fetch the embedding for.

        Returns:
            The embedding with the given document ID.
        """
        ...

    @abstractmethod
    def __contains__(self, document_id: Index) -> bool:
        """Check if a document exists in the store.

        Args:
            document_id:
                The ID of the document to check.

        Returns:
            Whether the document exists in the store.
        """
        ...

    @abstractmethod
    def __iter__(self) -> typing.Generator[Embedding, None, None]:
        """Iterate over the embeddings in the store.

        Yields:
            The embeddings in the store.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of embeddings in the store.

        Returns:
            The number of embeddings in the store.
        """
        ...

    def __repr__(self) -> str:
        """Return a string representation of the embedding store.

        Returns:
            A string representation of the embedding store.
        """
        return f"{self.__class__.__name__}({len(self):,} embeddings)"


class Generator(ABC):
    """An abstract generator of answers from a query and relevant documents."""

    stream: bool

    def compile(
        self,
        document_store: "DocumentStore",
        embedder: "Embedder",
        embedding_store: "EmbeddingStore",
    ) -> None:
        """Compile the embedder.

        Args:
            document_store:
                The document store to use.
            embedder:
                The embedder to use.
            embedding_store:
                The embedding store to use.
        """
        pass

    @abstractmethod
    def prompt_too_long(self, prompt: str) -> bool:
        """Check if a prompt is too long for the generator.

        Args:
            prompt:
                The prompt to check.

        Returns:
            Whether the prompt is too long for the generator.
        """
        ...

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

    def __repr__(self) -> str:
        """Return a string representation of the generator.

        Returns:
            A string representation of the generator.
        """
        return f"{self.__class__.__name__}()"


class PersistentSharingConfig(BaseModel):
    """The configuration for persistent sharing of a demo."""

    space_repo_id: str
    database_repo_id: str
    database_update_frequency: int
    hf_token_variable_name: str
