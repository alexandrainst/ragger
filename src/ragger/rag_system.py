"""The main entry point for the RAG system, orchestrating the other components."""

import logging
import typing

from .constants import DANISH_NO_DOCUMENTS_REPLY, ENGLISH_NO_DOCUMENTS_REPLY
from .data_models import (
    Document,
    DocumentStore,
    Embedder,
    EmbeddingStore,
    GeneratedAnswer,
    Generator,
)
from .document_store import JsonlDocumentStore
from .embedder import OpenAIEmbedder
from .embedding_store import NumpyEmbeddingStore
from .generator import OpenaiGenerator
from .utils import format_answer

logger = logging.getLogger(__package__)


class RagSystem:
    """The main entry point for the RAG system, orchestrating the other components."""

    def __init__(
        self,
        document_store: DocumentStore | None = None,
        embedder: Embedder | None = None,
        embedding_store: EmbeddingStore | None = None,
        generator: Generator | None = None,
        language: typing.Literal["da", "en"] = "da",
        no_documents_reply: str | None = None,
    ) -> None:
        """Initialise the RAG system.

        Args:
            document_store (optional):
                The document store to use, or None to use the default.
            embedder (optional):
                The embedder to use, or None to use the default.
            embedding_store (optional):
                The embedding store to use, or None to use the default.
            generator (optional):
                The generator to use, or None to use the default.
            language (optional):
                The language to use for the system. Can be "da" (Danish) or "en"
                (English). Defaults to "da".
            no_documents_reply (optional):
                The reply to use when no documents are found. If None, a default
                reply is used, based on the chosen language. Defaults to None.
        """
        # Use defaults if no components are provided
        if document_store is None:
            document_store = JsonlDocumentStore()
        if embedder is None:
            embedder = OpenAIEmbedder()
        if embedding_store is None:
            embedding_store = NumpyEmbeddingStore(embedding_dim=embedder.embedding_dim)
        if generator is None:
            generator = OpenaiGenerator(language=language)

        self.document_store = document_store
        self.embedder = embedder
        self.embedding_store = embedding_store
        self.generator = generator
        self.language = language

        no_documents_reply_mapping = dict(
            da=DANISH_NO_DOCUMENTS_REPLY, en=ENGLISH_NO_DOCUMENTS_REPLY
        )
        self.no_documents_reply = (
            no_documents_reply or no_documents_reply_mapping[language]
        )

        self.compile()

    @classmethod
    def from_config(cls, config: dict[str, typing.Any]) -> "RagSystem":
        """Create a RAG system from a configuration.

        Args:
            config:
                The configuration to create the system from.

        Returns:
            The created RAG system.
        """
        kwargs: dict[str, typing.Any] = dict()

        components = ["document_store", "embedder", "embedding_store", "generator"]
        for component in components:
            if component not in config:
                continue
            assert "name" in config[component], f"Missing 'name' key for {component}."
            module = __import__(name=component)
            component_class = getattr(module, config[component]["name"])
            kwargs[component] = component_class(**config[component])

        if "language" in config:
            kwargs["language"] = config["language"]
        if "no_documents_reply" in config:
            kwargs["no_documents_reply"] = config["no_documents_reply"]

        return cls(**kwargs)

    def compile(self, force: bool = False) -> "RagSystem":
        """Compile the RAG system.

        This builds the underlying embedding store and can be called whenever the data
        needs to be updated.

        Args:
            force:
                Whether to force a recompilation. This deletes the existing embedding
                store and rebuilds it.
        """
        if force:
            self.embedding_store.clear()

        self.document_store.compile(
            embedder=self.embedder,
            embedding_store=self.embedding_store,
            generator=self.generator,
        )
        self.embedder.compile(
            document_store=self.document_store,
            embedding_store=self.embedding_store,
            generator=self.generator,
        )
        self.embedding_store.compile(
            document_store=self.document_store,
            embedder=self.embedder,
            generator=self.generator,
        )
        self.generator.compile(
            document_store=self.document_store,
            embedder=self.embedder,
            embedding_store=self.embedding_store,
        )
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
                    source_documents = [
                        self.document_store[i]
                        for i in answer.sources
                        if i in self.document_store
                    ]
                    yield (answer.answer, source_documents)

            return streamer()
        else:
            source_documents = [
                self.document_store[i]
                for i in generated_answer.sources
                if i in self.document_store
            ]
            return (generated_answer.answer, source_documents)

    def answer_formatted(self, query: str) -> str | typing.Generator[str, None, None]:
        """Answer a query in a formatted single HTML string.

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
                        no_documents_reply=self.no_documents_reply,
                    )

            return streamer()
        answer, documents = output
        return format_answer(
            answer=answer,
            documents=documents,
            no_documents_reply=self.no_documents_reply,
        )

    def add_documents(
        self, documents: list[Document | str | dict[str, str]]
    ) -> "RagSystem":
        """Add documents to the store.

        Args:
            documents:
                An iterable of documents to add to the store
        """
        document_objects = [doc for doc in documents if isinstance(doc, Document)]
        string_documents = [doc for doc in documents if isinstance(doc, str)]
        dictionary_documents = [doc for doc in documents if isinstance(doc, dict)]

        # In case dictionaries have been passed, we convert them to documents
        for doc in dictionary_documents:
            if "text" not in doc:
                raise ValueError("The dictionary documents must have a 'text' key.")
            if "id" not in doc:
                string_documents.append(doc["text"])
            else:
                new_document = Document(id=doc["id"], text=doc["text"])
                document_objects.append(new_document)

        # In case raw strings have been passed, we find unused unique IDs for them
        new_idx = 0
        for text in string_documents:
            while str(new_idx) in self.document_store:
                new_idx += 1
            new_document = Document(id=str(new_idx), text=text)
            document_objects.append(new_document)
            new_idx += 1

        self.document_store.add_documents(documents=document_objects)
        embeddings = self.embedder.embed_documents(documents=document_objects)
        self.embedding_store.add_embeddings(embeddings=embeddings)
        return self

    def __repr__(self) -> str:
        """Return a string representation of the RAG system."""
        return (
            "RagSystem("
            f"document_store={self.document_store}, "
            f"embedder={self.embedder}, "
            f"embedding_store={self.embedding_store}, "
            f"generator={self.generator})"
        )
