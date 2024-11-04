"""Retrieve relevant documents for a given query."""

import logging
import os
import typing
from collections import defaultdict

from .data_models import (
    DocumentStore,
    Embedder,
    EmbeddingStore,
    Generator,
    Index,
    Retriever,
)
from .embedder import E5Embedder
from .embedding_store import NumpyEmbeddingStore
from .utils import is_installed, raise_if_not_installed

if is_installed(package_name="rank_bm25"):
    from rank_bm25 import BM25, BM25Okapi


if typing.TYPE_CHECKING:
    from rank_bm25 import BM25, BM25Okapi


os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__package__)


class EmbeddingRetriever(Retriever):
    """A retriever using an embedding model to retrieve relevant documents."""

    def __init__(
        self,
        embedder: Embedder = E5Embedder(),
        embedding_store: EmbeddingStore = NumpyEmbeddingStore(),
    ) -> None:
        """Initialise the E5 retriever.

        Args:
            embedder (optional):
                The embedder to use. Defaults to an E5 embedder.
            embedding_store (optional):
                The embedding store to use. Defaults to a NumPy embedding store.
        """
        self.embedder = embedder
        self.embedding_store = embedding_store

    def compile(self, document_store: "DocumentStore", generator: "Generator") -> None:
        """Compile the retriever.

        Args:
            document_store:
                The document store to use.
            generator:
                The generator to use.
        """
        self.embedder.compile(
            document_store=document_store, retriever=self, generator=generator
        )
        self.embedding_store.compile(
            document_store=document_store, retriever=self, generator=generator
        )

    def retrieve(self, query: str) -> list[Index]:
        """Retrieve relevant documents for a query.

        Args:
            query:
                The query to retrieve documents for.

        Returns:
            A list of document IDs.
        """
        query_embedding = self.embedder.embed_query(query=query)
        document_ids = self.embedding_store.get_nearest_neighbours(
            embedding=query_embedding
        )
        return document_ids

    def __repr__(self) -> str:
        """Return a string representation of the retriever."""
        return (
            f"{self.__class__.__name__}"
            f"(embedder={self.embedder}, embedding_store={self.embedding_store})"
        )


class BM25Retriever(Retriever):
    """A retriever using BM25 to retrieve relevant documents."""

    def __init__(
        self,
        tokenizer: typing.Callable[[str], list[str]] = lambda x: x.split(),
        preprocessor: typing.Callable[[str], str] = lambda x: x.lower(),
    ) -> None:
        """Initialise the BM25 retriever.

        Args:
            tokenizer:
                A function that tokenizes a text into a list of tokens. Defaults to
                splitting the text by whitespace.
            preprocessor:
                A function that preprocesses a text. Defaults to converting the text
                to lowercase.
        """
        raise_if_not_installed(package_names=["rank_bm25"], extra="keyword-search")
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.row_id_to_index: dict[int, Index] = defaultdict()
        self._bm25: BM25 | None = None

    def compile(self, document_store: "DocumentStore", generator: "Generator") -> None:
        """Compile the retriever.

        Args:
            document_store:
                The document store to use.
            generator:
                The generator to use.
        """
        corpus = [document for document in document_store]
        self.row_id_to_index = {
            row_id: document.id for row_id, document in enumerate(corpus)
        }
        tokenised_corpus = [
            self.tokenizer(self.preprocessor(doc.text)) for doc in corpus
        ]
        self._bm25 = BM25Okapi(corpus=tokenised_corpus)

    def retrieve(self, query: str) -> list[Index]:
        """Retrieve relevant documents for a query.

        Args:
            query:
                The query to retrieve documents for.

        Returns:
            A list of document IDs.
        """
        if self._bm25 is None:
            raise RuntimeError("The BM25 retriever has not been compiled.")

        tokenised_query = self.tokenizer(self.preprocessor(query))
        scores = self._bm25.get_scores(tokenised_query)
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [self.row_id_to_index[row_id] for row_id, _ in sorted_scores]


class HybridRetriever(Retriever):
    """A retriever that fuses rankings from multiple retrievers."""

    def __init__(
        self,
        retrievers: list[Retriever] = [EmbeddingRetriever(), BM25Retriever()],
        fusion_method: typing.Literal["reciprocal_rank"] = "reciprocal_rank",
    ) -> None:
        """Initialise the fuser retriever.

        Args:
            retrievers (optional):
                The retrievers to fuse. Defaults to an embedding retriever and a BM25
                retriever.
            fusion_method (optional):
                The method to use for fusing the rankings. Currently only supports
                'reciprocal_rank', being the method from [1]. Defaults to
                'reciprocal_rank'.

        References:
            [1] Cormack, Gordon V., Charles LA Clarke, and Stefan Buettcher. "Reciprocal
                rank fusion outperforms condorcet and individual rank learning methods."
                Proceedings of the 32nd international ACM SIGIR conference on Research
                and development in information retrieval. 2009.
        """
        self.retrievers = retrievers
        self.fusion_method = fusion_method

    def compile(self, document_store: "DocumentStore", generator: "Generator") -> None:
        """Compile the retriever.

        Args:
            document_store:
                The document store to use.
            generator:
                The generator to use.
        """
        for retriever in self.retrievers:
            retriever.compile(document_store=document_store, generator=generator)

    def retrieve(self, query: str) -> list[Index]:
        """Retrieve relevant documents for a query.

        Args:
            query:
                The query to retrieve documents for.

        Returns:
            A list of document IDs.
        """
        rankings = [retriever.retrieve(query=query) for retriever in self.retrievers]
        match self.fusion_method:
            case "reciprocal_rank":
                return self._reciprocal_rank_fusion(rankings=rankings)
            case _:
                raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

    @staticmethod
    def _reciprocal_rank_fusion(
        rankings: list[list[Index]], k: int = 60
    ) -> list[Index]:
        """Fuse rankings using reciprocal rank fusion.

        Args:
            rankings:
                A list of rankings, where each ranking is a list of document IDs.
            k:
                The hyperparameter `k` to use in the formula. Defaults to 60.

        Returns:
            The fused ranking.
        """
        scores: typing.DefaultDict[Index, float] = defaultdict(float)
        for ranking in rankings:
            for rank, document_index in enumerate(ranking):
                scores[document_index] += 1 / (k + rank)
        sorted_scores = sorted(list(scores.items()), key=lambda x: x[1], reverse=True)
        return [document_index for document_index, _ in sorted_scores]
