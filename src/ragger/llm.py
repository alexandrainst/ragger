"""Module for providing answer to the question."""

import logging
from typing import Generator

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def answer_question_with_qa(
    query: str, documents: list[str], cfg: DictConfig
) -> str | Generator[str, None, None]:
    """Write the answer using the relevant QA's.

    Args:
        query:
            A question.
        documents:
            The relevant documents.
        cfg:
            The Hydra configuration.

    Returns:
        The answer to the question.
    """
    # TODO: Get answer from documents using the QA model
    return "I don't know the answer to that question."
