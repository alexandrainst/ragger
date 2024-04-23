"""Embed documents using a pre-trained model."""

import logging
import os
import re
from abc import ABC, abstractmethod

import numpy as np
import torch
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

from .utils import Document

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """An abstract embedder, which embeds documents using a pre-trained model."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the embedder.

        Args:
            config:
                The Hydra configuration.
        """
        self.config = config

    @abstractmethod
    def embed_documents(self, documents: list[Document]) -> np.ndarray:
        """Embed a list of documents.

        Args:
            documents:
                A list of documents to embed.

        Returns:
            An array of embeddings, where each row corresponds to a document.
        """
        ...


class E5Embedder(Embedder):
    """An embedder that uses an E5 model to embed documents."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the E5 embedder.

        Args:
            config:
                The Hydra configuration.
        """
        super().__init__(config)
        self.embedder = SentenceTransformer(self.config.embedder_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def embed_documents(self, documents: list[Document]) -> np.ndarray:
        """Embed a list of documents using an E5 model.

        Args:
            documents:
                A list of documents to embed.

        Returns:
            An array of embeddings, where each row corresponds to a document.
        """
        # Prepare the texts for embedding
        texts = [document.text for document in documents]
        prepared_texts = self._prepare_texts_for_embedding(texts=texts)

        # Embed the texts
        embeddings = self.embedder.encode(
            sentences=prepared_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        assert isinstance(embeddings, np.ndarray)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query.

        Args:
            query:
                A query.

        Returns:
            The embedding of the query.
        """
        prepared_query = self._prepare_query_for_embedding(query=query)
        query_embedding = self.embedder.encode(
            sentences=[prepared_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device,
        )[0]
        return query_embedding

    def _prepare_texts_for_embedding(self, texts: list[str]) -> list[str]:
        """This prepares texts for embedding.

        The precise preparation depends on the embedding model and usecase.

        Args:
            texts:
                The texts to prepare.

        Returns:
            The prepared texts.
        """
        passages = [
            "passage: " + re.sub(r"^passage: ", "", passage) for passage in texts
        ]
        return passages

    def _prepare_query_for_embedding(self, query: str) -> str:
        """This prepares a query for embedding.

        The precise preparation depends on the embedding model.

        Args:
            query:
                A query.

        Returns:
            A prepared query.
        """
        query = "query: " + re.sub(r"^query: ", "", query)
        return query
