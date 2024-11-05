"""Embed documents using a pre-trained model."""

import logging
import os
import re
import typing
from functools import cache

import numpy as np

from .data_models import Document, Embedder, Embedding
from .utils import get_device, is_installed, raise_if_not_installed

if is_installed(package_name="openai"):
    from openai import OpenAI

if is_installed(package_name="sentence_transformers"):
    from sentence_transformers import SentenceTransformer

if is_installed(package_name="transformers"):
    from transformers import AutoConfig, AutoTokenizer

if is_installed(package_name="rank_bm25"):
    pass

if typing.TYPE_CHECKING:
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    from transformers import AutoConfig, AutoTokenizer


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
        raise_if_not_installed(
            package_names=["sentence_transformers"], extra=["onprem_cpu", "onprem_gpu"]
        )

        self.embedder_model_id = embedder_model_id
        self.device = get_device() if device == "auto" else device

        self.embedder = SentenceTransformer(
            model_name_or_path=self.embedder_model_id, device=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.embedder_model_id, token=os.getenv("HUGGINGFACE_HUB_TOKEN", True)
        )
        self.model_config = AutoConfig.from_pretrained(
            self.embedder_model_id, token=os.getenv("HUGGINGFACE_HUB_TOKEN", True)
        )
        self.embedding_dim = self.model_config.hidden_size

    def embed_documents(self, documents: typing.Iterable[Document]) -> list[Embedding]:
        """Embed a list of documents using an E5 model.

        Args:
            documents:
                An iterable of documents to embed.

        Returns:
            A list of embeddings, where each row corresponds to a document.

        Raises:
            RuntimeError:
                If the embedder has not been compiled.
        """
        if self.embedder is None:
            raise RuntimeError("The embedder has not been compiled.")

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
            RuntimeError:
                If the embedder has not been compiled.
        """
        if self.embedder is None:
            raise RuntimeError("The embedder has not been compiled.")

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


class OpenAIEmbedder(Embedder):
    """An embedder that uses an OpenAI model to embed documents."""

    def __init__(
        self,
        embedder_model_id: str = "text-embedding-3-small",
        api_key: str | None = None,
        host: str | None = None,
        port: int = 8000,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """Initialise the OpenAI embedder.

        Args:
            embedder_model_id (optional):
                The model ID of the embedder to use. Defaults to
                "text-embedding-3-small".
            api_key (optional):
                The API key to use. Defaults to None.
            host (optional):
                The host to use. Defaults to None.
            port (optional):
                The port to use. Defaults to 8000.
            timeout (optional):
                The timeout to use. Defaults to 60.
            max_retries (optional):
                The maximum number of retries. Defaults to 3.
        """
        raise_if_not_installed(package_names=["openai"], extra="openai")

        self.embedder_model_id = embedder_model_id
        self.api_key = api_key
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries

        # Set the server URL, if a host is provided
        self.server: str | None
        if self.host is not None:
            if not self.host.startswith("http"):
                self.host = f"http://{host}"
            self.server = f"{self.host}:{self.port}/v1"
        else:
            self.server = None

        self.client = OpenAI(
            base_url=self.server,
            api_key=api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self.embedding_dim = self._get_embedding_dim()

    @cache
    def _get_embedding_dim(self) -> int:
        """Return the embedding dimension of the OpenAI model."""
        embedding = self._embed(texts=["x"])
        return embedding.shape[1]

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using an OpenAI model.

        Args:
            texts:
                The texts to embed.

        Returns:
            The embeddings of the texts, of shape (n_texts, self.embedding_dim).
        """
        model_output = self.client.embeddings.create(
            input=texts, model=self.embedder_model_id
        )
        return np.stack(
            arrays=[np.asarray(embedding.embedding) for embedding in model_output.data],
            axis=0,
        )

    def embed_documents(self, documents: typing.Iterable[Document]) -> list[Embedding]:
        """Embed a list of documents.

        Args:
            documents:
                An iterable of documents to embed.

        Returns:
            An array of embeddings, where each row corresponds to a document.
        """
        if not documents:
            return list()

        raw_embeddings = self._embed(texts=[doc.text for doc in documents])

        return [
            Embedding(id=document.id, embedding=embedding)
            for document, embedding in zip(documents, raw_embeddings)
        ]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query.

        Args:
            query:
                A query.

        Returns:
            The embedding of the query.
        """
        return self._embed(texts=[query])[0]
