"""Embedding of documents."""

import io
import json
import logging
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig

from .utils import get_device

os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)


class Embedder:
    """Embedder class of documents.

    Args:
        cfg: The Hydra configuration.

    Attributes:
        cfg: The Hydra configuration.
        embedding_dim: The embedding dimension.
        documents: A dictionary of groups and their documents.
        embeddings: A dictionary of documents and their embeddings.
        device: The device to use for the embedding model.
        embedder: The SentenceTransformer object for embedding.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialises the Embedder object.

        Args:
            cfg:
                The Hydra configuration.
        """
        self.cfg = cfg
        self.embedding_dim = self._get_embedding_dimension()
        self.documents: dict[str, list[str]] = defaultdict(list)
        self.embeddings: dict[str, np.ndarray] = dict()
        self.device = get_device()
        self.embedder = SentenceTransformer(
            model_name_or_path=cfg.poc1.embedding_model.id, device=self.device
        )
        logger.info(f"Embedder initialized with {self.cfg.poc1.embedding_model.id}")

    def add_documents(self, documents: list[str], group: str = "default") -> "Embedder":
        """This adds documents to the embeddings store.

        This both updates the `documents` attribute with the raw documents and the
        `embeddings` attribute with the embeddings of the prepared versions of the
        documents.

        Args:
            documents:
                A list of documents.
            group:
                The group to add the documents to.

        Raises:
            TypeError: If `group` is not a string.
        """
        if not isinstance(group, str):
            raise TypeError(f"Expected group to be a string, got {type(group)}")

        documents = [
            document for document in documents if isinstance(document, str) and document
        ]

        prepared_documents = self._prepare_passages_for_embedding(passages=documents)
        embeddings = self.embedder.encode(
            sentences=prepared_documents,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device,
        )
        assert isinstance(embeddings, np.ndarray)
        self.documents[group].extend(documents)

        if group not in self.embeddings:
            self.embeddings[group] = embeddings
        else:
            self.embeddings[group] = np.vstack([self.embeddings[group], embeddings])

        assert len(self.documents[group]) == self.embeddings[group].shape[0]
        return self

    def get_relevant_documents(
        self, query: str, num_documents: int, groups: list[str] = ["default"]
    ) -> list[str]:
        """This returns the most relevant documents to the query.

        Args:
            query: A query.
            num_documents: The number of documents to return.
            groups: The groups to search for relevant documents.

        Returns:
            A list of documents.
        """
        unknown_groups: set[str] = set(groups) - set(self.documents.keys())
        if len(unknown_groups) == len(groups):
            logger.warning(
                f"None of the groups {groups} are known. Returning empty list."
            )
            return list()
        elif unknown_groups:
            logger.warning(f"Do not know groups {unknown_groups}. Removing them.")
            groups = list(set(groups).difference(unknown_groups))

        prepared_query = self._prepare_query_for_embedding(query=query)
        query_embedding = self.embedder.encode(
            sentences=[prepared_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device,
        )[0]

        # Get all the embeddings and documents for the groups
        embeddings = np.vstack([self.embeddings[group] for group in groups])
        documents = [document for group in groups for document in self.documents[group]]

        # Get the top-k documents
        scores = embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:num_documents]
        top_documents = [documents[i] for i in top_indices]
        top_documents_str = "\n\t-".join(top_documents)
        logger.info(
            f"Retrieved the following relevant questions:\n\t-{top_documents_str}"
        )
        return top_documents

    def reset(self) -> "Embedder":
        """This resets the embeddings store."""
        self.documents = defaultdict(list)
        self.embeddings = defaultdict(lambda: np.empty(shape=(self.embedding_dim,)))
        return self

    def save(self, path: Path | str) -> "Embedder":
        """This saves the embeddings store to disk.

        This will store the documents and the embeddings in a zip file, with the
        documents being stored in `documents.json` and the embeddings in
        `embeddings.npy` within the zip file.

        Args:
            path: The path to the zip file to store the embeddings store in.
        """
        path = Path(path)

        array_file = io.BytesIO()
        np.savez_compressed(file=array_file, **self.embeddings)

        documents_encoded = json.dumps(obj=self.documents).encode("utf-8")

        with zipfile.ZipFile(
            file=path, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as zf:
            zf.writestr("documents.json", data=documents_encoded)
            zf.writestr("embeddings.npz", data=array_file.getvalue())

        return self

    def load(self, path: Path | str) -> "Embedder":
        """This loads the embeddings store from disk.

        Args:
            path: The path to the zip file to load the embeddings store from.

        Returns:
            An Embedder object.
        """
        path = Path(path)
        with zipfile.ZipFile(file=path, mode="r") as zf:
            documents_encoded = zf.read("documents.json")
            documents = json.loads(documents_encoded.decode("utf-8"))
            array_file = io.BytesIO(zf.read("embeddings.npz"))
            embeddings = np.load(file=array_file, allow_pickle=False)
        self.documents = documents
        self.embeddings = embeddings
        return self

    def _prepare_passages_for_embedding(self, passages: list[str]) -> list[str]:
        """This prepares passages for embedding.

        The precise preparation depends on the embedding model.

        Args:
            passages: A list of documents.

        Returns:
            A list of prepared documents.
        """
        # Remove comments in the form of ＜...＞
        passages = [re.sub(r"＜.+＞", "", passage).strip() for passage in passages]

        # Remove enumeration prefixes
        passages = [re.sub(r"^\d+\.", "", passage).strip() for passage in passages]

        # Add question marks at the end of questions, if not already present
        passages = [re.sub(r"[。\?]$", "？", passage).strip() for passage in passages]
        passages = [
            passage + "？" if not passage.endswith("？") else passage
            for passage in passages
        ]

        if "-e5-" in self.cfg.poc1.embedding_model.id:
            passages = [
                "passage: " + re.sub(r"^passage: ", "", passage) for passage in passages
            ]

        return passages

    def _prepare_query_for_embedding(self, query: str) -> str:
        """This prepares a query for embedding.

        The precise preparation depends on the embedding model.

        Args:
            query: A query.

        Returns:
            A prepared query.
        """
        # Add question marks at the end of the question, if not already present
        query = re.sub(r"[。\?]$", "？", query).strip()
        if not query.endswith("？"):
            query += "？"

        if "-e5-" in self.cfg.poc1.embedding_model.id:
            query = "passage: " + re.sub(r"^passage: ", "", query)
        return query

    def _get_embedding_dimension(self) -> int:
        """This returns the embedding dimension for the embedding model.

        Returns:
            The embedding dimension.
        """
        model_config = AutoConfig.from_pretrained(self.cfg.poc1.embedding_model.id)
        return model_config.hidden_size
