"""The main entry point for the RAG system, orchestrating the other components."""

from omegaconf import DictConfig

from .document_store import DocumentStore, JsonlDocumentStore
from .embedder import E5Embedder, Embedder
from .embedding_store import EmbeddingStore, NumpyEmbeddingStore
from .generator import Generator, OpenAIGenerator
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

    def answer(self, query: str) -> tuple[str, list[Document]]:
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

        return (
            generated_answer.answer,
            [self.document_store[i] for i in generated_answer.sources],
        )
