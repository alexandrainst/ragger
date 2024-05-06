"""The main entry point for the RAG system, orchestrating the other components."""

import logging
import typing

from omegaconf import DictConfig

from .document_store import DocumentStore, JsonlDocumentStore
from .embedder import E5Embedder, Embedder
from .embedding_store import EmbeddingStore, NumpyEmbeddingStore
from .generator import Generator, OpenAIGenerator
from .utils import Document, GeneratedAnswer, format_answer

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

    def compile(self) -> None:
        """Compile the RAG system.

        This builds the underlying embedding store and can be called whenever the data
        needs to be updated.
        """
        match name := self.config.document_store.name:
            case "jsonl":
                self.document_store = JsonlDocumentStore(config=self.config)
            case _:
                raise ValueError(f"The DocumentStore type {name!r} is not supported")

        match name := self.config.embedder.name:
            case "e5":
                self.embedder = E5Embedder(config=self.config)
            case _:
                raise ValueError(f"The Embedder type {name!r} is not supported")

        match name := self.config.embedding_store.name:
            case "numpy":
                self.embedding_store = NumpyEmbeddingStore(config=self.config)
            case _:
                raise ValueError(f"The EmbeddingStore type {name!r} is not supported")

        match name := self.config.generator.name:
            case "openai":
                self.generator = OpenAIGenerator(config=self.config)
            case _:
                raise ValueError(f"The Generator type {name!r} is not supported")

        documents = self.document_store.get_all_documents()
        documents_not_in_embedding_store = [
            document
            for document in documents
            if document.id not in self.embedding_store.index_to_row_id
        ]
        embeddings = self.embedder.embed_documents(
            documents=documents_not_in_embedding_store
        )
        self.embedding_store.add_embeddings(embeddings=embeddings)

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
        query_embedding = self.embedder.embed_query(query)
        nearest_neighbours = self.embedding_store.get_nearest_neighbours(
            query_embedding
        )
        generated_answer = self.generator.generate(
            query=query, documents=[self.document_store[i] for i in nearest_neighbours]
        )
        if isinstance(generated_answer, typing.Generator):

            def streamer() -> typing.Generator[tuple[str, list[Document]], None, None]:
                answer = GeneratedAnswer(answer="")
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
