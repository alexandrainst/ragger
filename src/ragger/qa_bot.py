"""The main question-answering model, wrapping all the other modules."""

import logging
from collections import defaultdict
from typing import Generator

from omegaconf import DictConfig

from .embedder import Embedder
from .llm import answer_question_with_qa
from .types import History

logger = logging.getLogger(__name__)


class QABot:
    """The main question-answering model, wrapping all the other modules.

    Args:
        cfg:
            The configuration dictionary.

    Attributes:
        cfg:
            The configuration dictionary.
        embedder:
            The Embedder object.
        qas:
            A dictionary of questions and their answers.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialises the QABot object.

        Args:
            cfg:
                The configuration dictionary.
        """
        self.cfg = cfg
        self.embedder = Embedder(cfg=cfg)
        self.qas: dict[str, dict[str, str]] = defaultdict(dict)
        logger.info(f"QABot initialized, with {len(self.qas):,} groups.")

    def __call__(self, history: History) -> str | Generator[str, None, None]:
        """Answer a question given a context.

        Args:
            history:
                A list of messages in chronological order.

        Returns:
            The answer to the question.
        """
        question = history[-1][0]
        assert isinstance(question, str)

        # Retrieve the relevant documents from the FAQs
        relevant_documents: list[str] = [""]
        # relevant_documents = self.embedder.get_relevant_documents(
        #     query=question,
        #     num_documents=self.cfg.poc1.embedding_model.num_documents_to_retrieve,
        # )

        # If no relevant documents were found then simply return that no answer was
        # found
        if not relevant_documents:
            return (
                "Beklager, jeg kunne ikke finde et svar på dit spørgsmål. Kan du "
                "omformulere det?"
            )

        return answer_question_with_qa(
            query=question, documents=relevant_documents, cfg=self.cfg
        )
