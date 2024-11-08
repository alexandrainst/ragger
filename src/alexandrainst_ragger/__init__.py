"""A general-purpose RAG system and demo application."""

import importlib.metadata
import logging

from .data_models import (
    Document,
    DocumentStore,
    Embedder,
    Embedding,
    EmbeddingStore,
    Generator,
    Index,
    Retriever,
)
from .demo import Demo
from .document_store import (
    JsonlDocumentStore,
    PostgresDocumentStore,
    SqliteDocumentStore,
    TxtDocumentStore,
)
from .embedder import E5Embedder, OpenAIEmbedder
from .embedding_store import NumpyEmbeddingStore, PostgresEmbeddingStore
from .generator import GGUFGenerator, OpenAIGenerator, VllmGenerator
from .rag_system import RagSystem
from .retriever import BM25Retriever, EmbeddingRetriever, HybridRetriever
from .webui_utils import RaggerPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__ or "alexandrainst_ragger")
