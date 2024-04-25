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
        match type_ := self.config.document_store.type:
            case "jsonl":
                self.document_store = JsonlDocumentStore(config=self.config)
            case _:
                raise ValueError(f"The DocumentStore type {type_!r} is not supported")

        match type_ := self.config.embedder.type:
            case "e5":
                self.embedder = E5Embedder(config=self.config)
            case _:
                raise ValueError(f"The Embedder type {type_!r} is not supported")

        match type_ := self.config.embedding_store.type:
            case "numpy":
                self.embedding_store = NumpyEmbeddingStore(config=self.config)
            case _:
                raise ValueError(f"The EmbeddingStore type {type_!r} is not supported")

        match type_ := self.config.generator.type:
            case "openai":
                self.generator = OpenAIGenerator(config=self.config)
            case _:
                raise ValueError(f"The Generator type {type_!r} is not supported")

        documents = self.document_store.get_all_documents()
        embeddings = self.embedder.embed_documents(documents=documents)
        self.embedding_store.add_embeddings(embeddings=embeddings)
        logger.info("Finished compiling the RAG system")

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
        logger.info(f"User asked the question: {query!r}")
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
                logger.info(f"Generated the answer: {answer.answer!r}")

            return streamer()
        else:
            logger.info(f"Generated the answer: {generated_answer.answer!r}")
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
