"""Utility constants and functions used in the project."""

import re

from .data_models import Document


def format_answer(
    answer: str, documents: list[Document], no_documents_reply: str
) -> str:
    """Format the answer with the relevant documents.

    Args:
        answer:
            The generated answer.
        documents:
            The relevant documents.
        no_documents_reply:
            The reply to use when no documents are found.

    Returns:
        The formatted answer.
    """
    match len(documents):
        case 0:
            answer = no_documents_reply
        case 1:
            answer += "\n\nKilde:\n\n"
        case _:
            answer += "\n\nKilder:\n\n"

    formatted_ids = [
        f"<a href='{document.id}'>{document.id}</a>"
        if is_link(text=document.id)
        else document.id
        for document in documents
    ]

    answer += "\n\n".join(
        f"<details><summary>{formatted_id}</summary>{document.text}</details>"
        for formatted_id, document in zip(formatted_ids, documents)
    )
    return answer


def is_link(text: str) -> bool:
    """Check if the text is a link.

    Args:
        text:
            The text to check.

    Returns:
        Whether the text is a link.
    """
    url_regex = r"(https?:\/\/)?(\w+\.)+\w{2,4}(\/#?\w+)*?(\/\w+\.\w{1,4}){0,1}"
    return re.match(pattern=url_regex, string=text) is not None
