"""Embedding of documents."""

import logging
import os
import re
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

from .utils import get_device

os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)


class Embedder:
    """Embedder class of documents.

    Args:
        cfg:
            The Hydra configuration.

    Attributes:
        cfg:
            The Hydra configuration.
        device:
            The device to use for the embedding model.
        embedder:
            The SentenceTransformer object for embedding.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialises the Embedder object.

        Args:
            cfg:
                The Hydra configuration.
        """
        self.cfg = cfg
        self.device = get_device()
        self.embedder = SentenceTransformer(
            model_name_or_path=cfg.poc1.embedding_model.id, device=self.device
        )
        logger.info(f"Embedder initialized with {self.cfg.embedding_model.id}")

    def embed_documents(self, documents: list[dict[str, str]]) -> None:
        """Embed documents.

        Args:
            documents:
                A list of documents.
        """
        # Load the embeddings from disk if they exist
        fname = self.cfg.embedding_model_id.replace("/", "--") + ".zip"
        embeddings_path = Path(self.cfg.dirs.data) / self.cfg.dirs.processed / fname
        if embeddings_path.exists():
            logger.info(f"Loading embeddings from {embeddings_path}")
            self.embedder.load(path=embeddings_path)
            return

        # Prepare the texts for embedding
        texts = [document[self.cfg.document_text_field] for document in documents]
        prepared_texts = self._prepare_texts_for_embedding(texts=texts)

        # Embed the texts
        embeddings = self.embedder.encode(
            sentences=prepared_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device,
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
        return texts

    def _prepare_query_for_embedding(self, query: str) -> str:
        """This prepares a query for embedding.

        The precise preparation depends on the embedding model.

        Args:
            query:
                A query.

        Returns:
            A prepared query.
        """
        # Add question marks at the end of the question, if not already present
        query = re.sub(r"[。\?]$", "？", query).strip()
        if not query.endswith("？"):
            query += "？"

        return query
