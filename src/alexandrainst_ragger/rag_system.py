"""The main entry point for the RAG system, orchestrating the other components."""

import logging
import typing
from pathlib import Path

from . import document_store as document_store_module
from . import embedder as embedder_module
from . import embedding_store as embedding_store_module
from . import generator as generator_module
from . import retriever as retriever_module
from .constants import DANISH_NO_DOCUMENTS_REPLY, ENGLISH_NO_DOCUMENTS_REPLY
from .data_models import Document, DocumentStore, GeneratedAnswer, Generator, Retriever
from .document_store import JsonlDocumentStore
from .generator import OpenAIGenerator
from .retriever import EmbeddingRetriever
from .utils import format_answer, load_config

logger = logging.getLogger(__package__)


class RagSystem:
    """The main entry point for the RAG system, orchestrating the other components."""

    def __init__(
        self,
        document_store: DocumentStore | None = None,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
        language: typing.Literal["da", "en"] = "da",
        no_documents_reply: str | None = None,
    ) -> None:
        """Initialise the RAG system.

        Args:
            document_store (optional):
                The document store to use, or None to use the default.
            retriever (optional):
                The retriever to use, or None to use the default. Defaults to None.
            generator (optional):
                The generator to use, or None to use the default. Defaults to None.
            language (optional):
                The language to use for the system. Can be "da" (Danish) or "en"
                (English). Defaults to "da".
            no_documents_reply (optional):
                The reply to use when no documents are found. If None, a default
                reply is used, based on the chosen language. Defaults to None.
        """
        self.document_store: DocumentStore = (
            JsonlDocumentStore() if document_store is None else document_store
        )
        self.retriever: Retriever = (
            EmbeddingRetriever() if retriever is None else retriever
        )
        self.generator: Generator = (
            OpenAIGenerator(language=language) if generator is None else generator
        )
        self.language = language

        no_documents_reply_mapping = dict(
            da=DANISH_NO_DOCUMENTS_REPLY, en=ENGLISH_NO_DOCUMENTS_REPLY
        )
        self.no_documents_reply = (
            no_documents_reply or no_documents_reply_mapping[language]
        )

        self.compile()

    @classmethod
    def from_config(cls, config_file: str | Path | None = None) -> "RagSystem":
        """Create a RAG system from a configuration.

        Args:
            config_file:
                The path to the configuration file, which should be a JSON or YAML file.

        Returns:
            The created RAG system.
        """
        config = load_config(config_file=config_file)

        kwargs: dict[str, typing.Any] = dict()

        components = dict(
            document_store=document_store_module,
            retriever=retriever_module,
            embedder=embedder_module,
            embedding_store=embedding_store_module,
            generator=generator_module,
        )
        for component, module in components.items():
            if component not in config:
                continue
            assert "name" in config[component], f"Missing 'name' key for {component}."
            component_class = getattr(module, config[component]["name"])
            config[component].pop("name")

            for key in config[component].keys():
                if key in components:
                    sub_component_class = getattr(
                        components[key], config[component][key]["name"]
                    )
                    config[component][key].pop("name")
                    config[component][key] = sub_component_class(
                        **config[component][key]
                    )

            kwargs[component] = component_class(**config[component])

        if "language" in config:
            kwargs["language"] = config["language"]
        if "no_documents_reply" in config:
            kwargs["no_documents_reply"] = config["no_documents_reply"]

        return cls(**kwargs)

    def compile(self) -> "RagSystem":
        """Compile the RAG system."""
        self.document_store.compile(retriever=self.retriever, generator=self.generator)
        self.retriever.compile(
            document_store=self.document_store, generator=self.generator
        )
        self.generator.compile(
            document_store=self.document_store, retriever=self.retriever
        )
        return self

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
        document_ids = self.retriever.retrieve(query=query)
        documents = [self.document_store[i] for i in document_ids]
        generated_answer = self.generator.generate(query=query, documents=documents)

        if isinstance(generated_answer, typing.Generator):

            def streamer() -> typing.Generator[tuple[str, list[Document]], None, None]:
                answer = GeneratedAnswer(answer="", sources=[])
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
        self.compile()
        return self

    def __repr__(self) -> str:
        """Return a string representation of the RAG system."""
        return (
            "RagSystem("
            f"document_store={self.document_store}, "
            f"retriever={self.retriever}, "
            f"generator={self.generator})"
        )
