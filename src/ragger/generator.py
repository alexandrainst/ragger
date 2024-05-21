"""Generation of an answer from a query and a list of relevant documents."""

import importlib.util
import json
import logging
import os
import typing

import torch
from dotenv import load_dotenv
from jinja2 import TemplateError
from omegaconf import DictConfig
from openai import OpenAI, Stream
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import ValidationError
from pydantic_core import from_json

from .data_models import Document, GeneratedAnswer, Generator
from .utils import clear_memory

if importlib.util.find_spec("vllm") is not None:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.guided_decoding.outlines_logits_processors import (
        JSONLogitsProcessor,
    )

    try:
        from vllm.model_executor.parallel_utils.parallel_state import (
            destroy_model_parallel,
        )
    except ImportError:
        from vllm.distributed.parallel_state import destroy_model_parallel


load_dotenv()


logger = logging.getLogger(__package__)


class OpenAIGenerator(Generator):
    """A generator that uses an OpenAI model to generate answers."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the OpenAI generator.

        Args:
            config:
                The Hydra configuration.
        """
        super().__init__(config=config)
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
        logger.info(
            f"Generating answer for the query {query!r} and {len(documents):,} "
            "documents..."
        )
        messages = [
            ChatCompletionSystemMessageParam(
                role="system", content=self.config.generator.system_prompt
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=self.config.generator.prompt.format(
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
            stop=["</answer>"],
            response_format=ResponseFormat(type="json_object"),
        )
        if isinstance(model_output, Stream):

            def streamer() -> typing.Generator[GeneratedAnswer, None, None]:
                generated_output = ""
                generated_obj = GeneratedAnswer(sources=[])
                for chunk in model_output:
                    chunk_str = chunk.choices[0].delta.content
                    if chunk_str is None:
                        break
                    generated_output += chunk_str
                    try:
                        generated_dict = from_json(
                            data=generated_output, allow_partial=True
                        )

                        # If the sources in the generated JSON dict is empty, but the
                        # final closing square bracket hasn't been generated yet, this
                        # means that the `from_json` function has closed this off
                        # itself, which is not allowed here, as this would trigger the
                        # "cannot answer" answer. To prevent this, we check for this
                        # and skip the next chunk if this is the case.
                        first_source_not_generated_yet = (
                            "sources" in generated_dict
                            and not generated_dict["sources"]
                            and '"sources": []' not in generated_output
                        )
                        if first_source_not_generated_yet:
                            continue

                        # If the answer is being written, the JSON dict will look like
                        #   '{"sources": [...], "answer": "Some text'
                        # As the answer doesn't have a closing quote, the `from_json`
                        # function will not include the `answer` key in the resulting
                        # dict. To ensure that the partial answer *is* included in the
                        # dict, we check if the model is currently writing the answer
                        # and if so, we add a closing quote to the generated output
                        # before attempting to parse it.
                        answer_partially_generated = (
                            "answer" not in generated_dict
                            and '"answer"' in generated_output
                        )
                        if answer_partially_generated:
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

        logger.info(f"Generated answer: {generated_obj.answer!r}")
        return generated_obj


class VLLMGenerator(Generator):
    """A generator that uses a vLLM model to generate answers."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the vLLM generator.

        Args:
            config:
                The Hydra configuration.
        """
        super().__init__(config=config)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "The `vLLMGenerator` requires a CUDA-compatible GPU to run. "
                "Please ensure that a compatible GPU is available and try again."
            )

        # We need to remove the model from GPU memory before creating a new one
        destroy_model_parallel()
        clear_memory()

        self.model = LLM(
            model=config.generator.model,
            gpu_memory_utilization=config.generator.gpu_memory_utilization,
            max_model_len=config.generator.max_model_len,
            seed=config.random_seed,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        self.tokenizer = self.model.get_tokenizer()
        self.logits_processor = JSONLogitsProcessor(
            schema=GeneratedAnswer, tokenizer=self.tokenizer, whitespace_pattern=r" ?"
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
        logger.info(
            f"Generating answer for the query {query!r} and {len(documents):,} "
            "documents..."
        )

        system_prompt = self.config.generator.system_prompt
        user_prompt = self.config.generator.prompt.format(
            documents=json.dumps([document.model_dump() for document in documents]),
            query=query,
        )

        chat_template_kwargs = dict(
            chat_template=self.tokenizer.chat_template,
            add_generation_prompt=True,
            tokenize=False,
        )
        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation=[
                    dict(role="system", content=system_prompt),
                    dict(role="user", content=user_prompt),
                ],
                **chat_template_kwargs,
            )
        except TemplateError:
            prompt = self.tokenizer.apply_chat_template(
                conversation=[
                    dict(role="user", content=system_prompt + "\n\n" + user_prompt)
                ],
                **chat_template_kwargs,
            )

        sampling_params = SamplingParams(
            max_tokens=self.config.generator.max_tokens,
            temperature=self.config.generator.temperature,
            stop=["</answer>"],
            logits_processors=[self.logits_processor],
        )

        model_output = self.model.generate(
            prompts=[prompt], sampling_params=sampling_params
        )
        generated_output = model_output[0].outputs[0].text

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

        logger.info(f"Generated answer: {generated_obj.answer!r}")
        return generated_obj
