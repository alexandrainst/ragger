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
from .rag_system import RagSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

# Fetches the version of the package as defined in pyproject.toml
__version__ = importlib.metadata.version(__package__ or "alexandrainst_ragger")
