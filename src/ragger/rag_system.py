"""The main entry point for the RAG system, orchestrating the other components."""

import logging
import typing
from pathlib import Path

from omegaconf import DictConfig

from .data_models import Document, GeneratedAnswer
from .document_store import DocumentStore
from .embedder import Embedder
from .embedding_store import EmbeddingStore
from .generator import Generator
from .utils import format_answer, load_ragger_components

logger = logging.getLogger(__package__)


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

    def compile(self, force: bool = False) -> "RagSystem":
        """Compile the RAG system.

        This builds the underlying embedding store and can be called whenever the data
        needs to be updated.

        Args:
            force:
                Whether to force a recompilation. This deletes the existing embedding
                store and rebuilds it.
        """
        components = load_ragger_components(config=self.config)
        for component_name, component_class in components.items():
            if not hasattr(self, component_name) or force:
                setattr(self, component_name, component_class(config=self.config))

        if force:
            embedding_store_path = (
                Path(self.config.dirs.data)
                / self.config.dirs.processed
                / self.config.embedding_store.filename
            )
            embedding_store_path.unlink(missing_ok=True)

        documents = self.document_store.get_all_documents()
        documents_not_in_embedding_store = [
            document
            for document in documents
            if not self.embedding_store.document_exists_in_store(document.id)
        ]

        embeddings = self.embedder.embed_documents(
            documents=documents_not_in_embedding_store
        )
        self.embedding_store.add_embeddings(embeddings=embeddings)
        return self

    def get_relevant_documents(self, query: str) -> list[Document]:
        """Get the most relevant documents for a query.

        Args:
            query:
                The query to find relevant documents for.

        Returns:
            The most relevant documents.
        """
        query_embedding = self.embedder.embed_query(query)
        nearest_neighbours = self.embedding_store.get_nearest_neighbours(
            query_embedding
        )
        return [self.document_store[i] for i in nearest_neighbours]

    def answer(
        self, query: str
    ) -> (
        tuple[str, list[Document]]
        | typing.Generator[tuple[str, list[Document]], None, None]
    ):
        """Answer a query.

        Args:
            query:
                The query to answer.

        Returns:
            A tuple of the answer and the supporting documents.
        """
        documents = self.get_relevant_documents(query=query)
        generated_answer = self.generator.generate(query=query, documents=documents)
        if isinstance(generated_answer, typing.Generator):

            def streamer() -> typing.Generator[tuple[str, list[Document]], None, None]:
                answer = GeneratedAnswer(sources=[])
                for answer in generated_answer:
                    assert isinstance(answer, GeneratedAnswer)
                    yield (
                        answer.answer,
                        [self.document_store[i] for i in answer.sources],
                    )

            return streamer()
        else:
            return (
                generated_answer.answer,
                [self.document_store[i] for i in generated_answer.sources],
            )

    def answer_formatted(self, query: str) -> str | typing.Generator[str, None, None]:
        """Answer a query in a formatted single string.

        The string includes both the answer and the supporting documents.

        Args:
            query:
                The query to answer.

        Returns:
            The formatted answer.
        """
        output = self.answer(query)
        if isinstance(output, typing.Generator):

            def streamer() -> typing.Generator[str, None, None]:
                for answer, documents in output:
                    assert isinstance(answer, str)
                    assert isinstance(documents, list)
                    yield format_answer(
                        answer=answer,
                        documents=documents,
                        no_documents_reply=self.config.demo.no_documents_reply,
                    )

            return streamer()
        answer, documents = output
        return format_answer(
            answer=answer,
            documents=documents,
            no_documents_reply=self.config.demo.no_documents_reply,
        )
