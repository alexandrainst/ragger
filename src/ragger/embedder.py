"""Embed documents using a pre-trained model."""

import logging
import os
import re
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoTokenizer

from .data_models import Document, Embedder, Embedding
from .utils import get_device

os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__package__)


class E5Embedder(Embedder):
    """An embedder that uses an E5 model to embed documents."""

    def __init__(
        self,
        embedder_model_id: str = "intfloat/multilingual-e5-large",
        device: str = "auto",
    ) -> None:
        """Initialise the E5 embedder.

        Args:
            embedder_model_id (optional):
                The model ID of the embedder to use. Defaults to
                "intfloat/multilingual-e5-large".
            device (optional):
                The device to use. If "auto", the device is chosen automatically based
                on hardware availability. Defaults to "auto".
        """
        self.embedder_model_id = embedder_model_id
        self.device = get_device() if device == "auto" else device

        self.embedder = SentenceTransformer(
            model_name_or_path=self.embedder_model_id, device=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedder_model_id)
        self.model_config = AutoConfig.from_pretrained(self.embedder_model_id)
        self.embedding_dim = self.model_config.hidden_size

    @property
    def max_context_length(self) -> int:
        """The maximum length of the context that the embedder can handle."""
        return self.tokenizer.model_max_length

    def tokenize(self, text: str | list[str]) -> np.ndarray:
        """Tokenize a text.

        Args:
            text:
                The text or texts to tokenize.

        Returns:
            The tokens of the text.
        """
        return self.tokenizer.tokenize(text)

    def embed_documents(self, documents: Iterable[Document]) -> list[Embedding]:
        """Embed a list of documents using an E5 model.

        Args:
            documents:
                An iterable of documents to embed.

        Returns:
            A list of embeddings, where each row corresponds to a document.

        Raises:
            ValueError:
                If the embedder has not been compiled.
        """
        if self.embedder is None:
            raise ValueError("The embedder has not been compiled.")

        if not documents:
            return list()

        # Prepare the texts for embedding
        texts = [document.text for document in documents]
        prepared_texts = self._prepare_texts_for_embedding(texts=texts)

        logger.info(
            f"Embedding {len(prepared_texts):,} documents with the E5 model "
            f"{self.embedder_model_id}..."
        )

        # Embed the texts
        assert self.embedder is not None
        embedding_matrix = self.embedder.encode(
            sentences=prepared_texts, normalize_embeddings=True, convert_to_numpy=True
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

        Raises:
            ValueError:
                If the embedder has not been compiled.
        """
        if self.embedder is None:
            raise ValueError("The embedder has not been compiled.")

        logger.info(f"Embedding the query {query!r} with the E5 model...")
        prepared_query = self._prepare_query_for_embedding(query=query)

        assert self.embedder is not None
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
