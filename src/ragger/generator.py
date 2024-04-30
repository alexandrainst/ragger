"""Generation of an answer from a query and a list of relevant documents."""

import json
import logging
import os
import typing
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from omegaconf import DictConfig
from openai import OpenAI, Stream
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import ValidationError
from pydantic_core import from_json

from .prompts import RAG_ANSWER_PROMPT, RAG_SYSTEM_PROMPT
from .utils import Document, GeneratedAnswer

load_dotenv()


logger = logging.getLogger(__package__)


class Generator(ABC):
    """An abstract generator of answers from a query and relevant documents."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the generator.

        Args:
            config:
                The Hydra configuration.
        """
        self.config = config

    @abstractmethod
    def generate(
        self, query: str, documents: list[Document]
    ) -> GeneratedAnswer | typing.Generator[GeneratedAnswer, None, None]:
        """Generate an answer from a query and relevant documents.

        Args:
            query:
                The query to answer.
            documents:
                The relevant documents.

        Returns:
            The generated answer.
        """
        ...


class OpenAIGenerator(Generator):
    """A generator that uses an OpenAI model to generate answers."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the OpenAI generator.

        Args:
            config:
                The Hydra configuration.
        """
        super().__init__(config)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)
        api_key = os.environ[self.config.generator.api_key_variable_name]
        self.client = OpenAI(
            api_key=api_key.strip('"'), timeout=self.config.generator.timeout
        )

    def generate(
        self, query: str, documents: list[Document]
    ) -> GeneratedAnswer | typing.Generator[GeneratedAnswer, None, None]:
        """Generate an answer from a query and relevant documents.

        Args:
            query:
                The query to answer.
            documents:
                The relevant documents.

        Returns:
            The generated answer.
        """
        messages = [
            ChatCompletionSystemMessageParam(role="system", content=RAG_SYSTEM_PROMPT),
            ChatCompletionUserMessageParam(
                role="user",
                content=RAG_ANSWER_PROMPT.format(
                    documents=json.dumps(
                        [document.model_dump() for document in documents]
                    ),
                    query=query,
                ),
            ),
        ]
        model_output = self.client.chat.completions.create(
            messages=messages,
            model=self.config.generator.model,
            max_tokens=self.config.generator.max_tokens,
            temperature=self.config.generator.temperature,
            stream=self.config.generator.stream,
            stop=["</svar>"],
            response_format=ResponseFormat(type="json_object"),
        )
        if isinstance(model_output, Stream):

            def streamer() -> typing.Generator[GeneratedAnswer, None, None]:
                generated_output = ""
                for chunk in model_output:
                    chunk_str = chunk.choices[0].delta.content
                    if chunk_str is None:
                        break
                    generated_output += chunk_str
                    try:
                        generated_dict = from_json(
                            data=generated_output, allow_partial=True
                        )
                        if "answer" not in generated_dict:
                            generated_dict = from_json(
                                data=generated_output + '"', allow_partial=True
                            )
                    except ValueError:
                        continue
                    try:
                        generated_obj = GeneratedAnswer.model_validate(generated_dict)
                        yield generated_obj
                    except ValidationError:
                        continue

            return streamer()
        else:
            generated_output = model_output.choices[0].message.content.strip()

        try:
            generated_dict = json.loads(generated_output)
        except json.JSONDecodeError:
            raise ValueError(
                f"Could not decode JSON from model output: {generated_output}"
            )

        try:
            generated_obj = GeneratedAnswer.model_validate(generated_dict)
        except ValidationError:
            raise ValueError(f"Could not validate model output: {generated_dict}")

        return generated_obj
