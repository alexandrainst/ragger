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
        if self.config.document_store.type == "jsonl":
            self.document_store = JsonlDocumentStore(self.config)
        else:
            raise ValueError("DocumentStore type not supported")

        if self.config.embedder.type == "e5":
            self.embedder = E5Embedder(self.config)
        else:
            raise ValueError("Embedder type not supported")

        if self.config.embedding_store.type == "numpy":
            self.embedding_store = NumpyEmbeddingStore(self.config)
        else:
            raise ValueError("EmbeddingStore store type not supported")

        # Get all documents from the document store
        documents = self.document_store.get_all_documents()

        # Embed all documents
        embeddings = self.embedder.embed_documents(documents)

        # Add all embeddings to the embedding store
        self.embedding_store.add_embeddings(embeddings)

    def answer(self, query: str) -> tuple[str, list[Document]]:
        """Answer a query.

        Args:
            query:
                The query to answer.

        Returns:
            A tuple of the answer and the supporting documents.
        """
        if self.config.generator.type == "openai":
            self.generator = OpenAIGenerator(self.config)
        else:
            raise ValueError("Generator type not supported")

        # Embed the query
        query_embedding = self.embedder.embed_query(query)

        # Get the nearest neighbours to the query embedding
        nearest_neighbours = self.embedding_store.get_nearest_neighbours(
            query_embedding
        )

        # Get the supporting documents
        supporting_documents = [self.document_store[i] for i in nearest_neighbours]

        # Generate the answer
        # generated_answer = self.generator.generate_answer(query, supporting_documents)

        # Extract documents from the generated answer source indices
        return ("bla", supporting_documents)
        #   return (
        #       generated_answer.answer,
        #       [supporting_documents[i] for i in generated_answer.source_indices],
        #   )
