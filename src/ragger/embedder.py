"""Embed documents using a pre-trained model."""

from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig

from .utils import Document


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

    pass
