"""Pipeline class that can be used to integrate with Open WebUI."""

import collections.abc as c
from abc import abstractmethod

from .constants import DANISH_NO_DOCUMENTS_REPLY, ENGLISH_NO_DOCUMENTS_REPLY
from .rag_system import RagSystem


class RaggerPipeline:
    """An abstract RAG pipeline using the Ragger package."""

    def __init__(self):
        """Initialise the pipeline."""
        self.rag_system: RagSystem | None = None

    @abstractmethod
    async def on_startup(self):
        """Run on startup."""
        ...

    def pipe(
        self, user_message: str, model_id: str, messages: list[dict], body: dict
    ) -> str | c.Generator | c.Iterator:
        """Run the pipeline.

        Args:
            user_message:
                The user message.
            model_id:
                The model ID.
            messages:
                The messages up to this point.
            body:
                The body.
        """
        assert self.rag_system is not None

        answer, sources = self.rag_system.answer(query=user_message)
        assert isinstance(answer, str) and isinstance(sources, list)

        if not answer or not sources:
            if self.rag_system.language == "da":
                return DANISH_NO_DOCUMENTS_REPLY
            else:
                return ENGLISH_NO_DOCUMENTS_REPLY

        formatted_sources = "\n".join(
            f"- **{source.id}**\n{source.text}" for source in sources
        )
        return f"{answer}\n\n### Kilder:\n{formatted_sources}"
