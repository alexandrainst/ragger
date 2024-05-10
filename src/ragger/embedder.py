"""Embed documents using a pre-trained model."""

import logging
import os
import re
from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig

from .data_models import Document, Embedding

os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__package__)


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
    def embed_documents(self, documents: list[Document]) -> list[Embedding]:
        """Embed a list of documents.

        Args:
            documents:
                A list of documents to embed.

        Returns:
            An array of embeddings, where each row corresponds to a document.
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query.

        Args:
            query:
                A query.

        Returns:
            The embedding of the query.
        """
        ...

    @abstractmethod
    def tokenize(self, text: str | list[str]) -> np.array:
        """Tokenize a text.

        Args:
            text:
                The text or texts to tokenize.

        Returns:
            The tokens of the text.
        """
        ...

    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """The maximum length of the context that the embedder can handle."""
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
        self.embedder = SentenceTransformer(
            self.config.embedder.model_id, cache_folder=self.config.dirs.models
        )

    @property
    def max_context_length(self) -> int:
        """The maximum length of the context that the embedder can handle."""
        embedder_config = AutoConfig.from_pretrained(self.config.embedder.model_id)
        return embedder_config.max_position_embeddings

    def tokenize(self, text: str | list[str]) -> np.array:
        """Tokenize a text.

        Args:
            text:
                The text or texts to tokenize.

        Returns:
            The tokens of the text.
        """
        return self.embedder.tokenize(text)

    def embed_documents(self, documents: list[Document]) -> list[Embedding]:
        """Embed a list of documents using an E5 model.

        Args:
            documents:
                A list of documents to embed.

        Returns:
            A list of embeddings, where each row corresponds to a document.
        """
        if not documents:
            return []

        logger.info(
            f"Building embeddings of {len(documents):,} documents with the E5 "
            f"model {self.config.embedder.model_id}..."
        )

        # Prepare the texts for embedding
        texts = [document.text for document in documents]
        prepared_texts = self._prepare_texts_for_embedding(texts=texts)

        # Embed the texts
        embedding_matrix = self.embedder.encode(
            sentences=prepared_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=self.config.verbose,
        )
        assert isinstance(embedding_matrix, np.ndarray)
        embeddings = [
            Embedding(id=document.id, embedding=embedding)
            for document, embedding in zip(documents, embedding_matrix)
        ]
        logger.info("Finished building embeddings.")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query.

        Args:
            query:
                A query.

        Returns:
            The embedding of the query.
        """
        logger.info(f"Embedding the query '{query}' with the E5 model...")
        prepared_query = self._prepare_query_for_embedding(query=query)
        query_embedding = self.embedder.encode(
            sentences=[prepared_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        assert isinstance(query_embedding, np.ndarray)
        logger.info("Finished embedding the query.")
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
