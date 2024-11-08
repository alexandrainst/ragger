"""Pipeline class that can be used to integrate with Open WebUI."""

import collections.abc as c
import logging
import os
from abc import abstractmethod

from huggingface_hub import login
from pydantic import BaseModel

from .constants import (
    DANISH_NO_DOCUMENTS_REPLY,
    DANISH_SOURCES,
    ENGLISH_NO_DOCUMENTS_REPLY,
)
from .generator import OpenAIGenerator
from .rag_system import RagSystem

logger = logging.getLogger(__package__)


class RaggerPipeline:
    """An abstract RAG pipeline using the Ragger package."""

    class Valves(BaseModel):
        """Configuration for the pipeline."""

        OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
        HUGGINGFACE_HUB_TOKEN: str | None = os.getenv("HUGGINGFACE_HUB_TOKEN")

    def __init__(self):
        """Initialise the pipeline."""
        self.rag_system: RagSystem | None = None
        self.valves = self.Valves()

    def update_valves(self) -> None:
        """Run when valves are updated."""
        if self.rag_system is None:
            return

        if self.valves.OPENAI_API_KEY is not None and isinstance(
            self.rag_system.generator, OpenAIGenerator
        ):
            self.rag_system.generator.api_key = self.valves.OPENAI_API_KEY

        if self.valves.HUGGINGFACE_HUB_TOKEN is not None:
            login(self.valves.HUGGINGFACE_HUB_TOKEN)

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

        if "stream" in body:
            self.rag_system.generator.stream = body["stream"]

        output = self.rag_system.answer(query=user_message)

        if isinstance(output, tuple):
            answer, sources = output

            if not answer or not sources:
                if self.rag_system.language == "da":
                    return DANISH_NO_DOCUMENTS_REPLY
                else:
                    return ENGLISH_NO_DOCUMENTS_REPLY

            sources_str = (
                DANISH_SOURCES if self.rag_system.language == "da" else "Sources"
            )
            formatted_sources = "\n".join(
                f"- **{source.id}**\n{source.text}" for source in sources
            )
            return f"{answer}\n\n### {sources_str}:\n{formatted_sources}"

        def generate():
            assert self.rag_system is not None
            for answer, sources in output:
                if not answer or not sources:
                    if self.rag_system.language == "da":
                        yield DANISH_NO_DOCUMENTS_REPLY
                    else:
                        yield ENGLISH_NO_DOCUMENTS_REPLY
                    continue

                formatted_sources = "\n".join(
                    f"- **{source.id}**\n{source.text}" for source in sources
                )
                yield f"{answer}\n\n### Kilder:\n{formatted_sources}"

        return generate()
