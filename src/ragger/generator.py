"""Generation of an answer from a query and a list of relevant documents."""

import json
import logging
import os
import subprocess
import typing
from time import sleep

import tiktoken
import torch
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
from transformers import AutoTokenizer

from .data_models import Document, GeneratedAnswer, Generator

load_dotenv()


logger = logging.getLogger(__package__)


class OpenaiGenerator(Generator):
    """A generator that uses an OpenAI model to generate answers."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the OpenAI generator.

        Args:
            config:
                The Hydra configuration.
        """
        super().__init__(config=config)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

        api_key = "not-set"
        if hasattr(config.generator, "api_key_variable_name"):
            env_var_name = config.generator.api_key_variable_name
            api_key_from_variable = os.environ[env_var_name].strip('"')
            if api_key_from_variable:
                api_key = api_key_from_variable

        self.server: str | None
        if hasattr(config.generator, "server"):
            host = config.generator.server
            if not host.startswith("http"):
                host = f"http://{host}"
            self.server = f"{host}:{config.generator.port}/v1"
        else:
            self.server = None

        self.client = OpenAI(
            base_url=self.server,
            api_key=api_key,
            timeout=self.config.generator.timeout,
            max_retries=self.config.generator.max_retries,
        )

    def prompt_too_long(self, prompt: str) -> bool:
        """Check if a prompt is too long for the generator.

        Args:
            prompt:
                The prompt to check.

        Returns:
            Whether the prompt is too long for the generator.
        """
        encoding = tiktoken.encoding_for_model(model_name=self.config.generator.model)
        num_tokens = len(encoding.encode(text=prompt))
        return num_tokens > self.config.generator.max_input_tokens

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
        for num_documents_to_include in range(len(documents), -1, -1):
            logger.info(
                f"Generating answer for the query {query!r} and "
                f"{num_documents_to_include:,} documents..."
            )
            messages = [
                ChatCompletionSystemMessageParam(
                    role="system", content=self.config.generator.system_prompt
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=self.config.generator.prompt.format(
                        documents=json.dumps(
                            [
                                document.model_dump()
                                for document in documents[:num_documents_to_include]
                            ]
                        ),
                        query=query,
                    ),
                ),
            ]
            if not self.prompt_too_long(prompt=json.dumps(messages)):
                break
        else:
            return GeneratedAnswer(sources=[], answer="Prompt too long.")

        extra_body = dict()
        if self.config.generator.name == "vllm":
            extra_body["guided_json"] = GeneratedAnswer.model_json_schema()

        model_output = self.client.chat.completions.create(
            messages=messages,
            model=self.config.generator.model,
            max_tokens=self.config.generator.max_output_tokens,
            temperature=self.config.generator.temperature,
            stream=self.config.generator.stream,
            stop=["</answer>"],
            response_format=ResponseFormat(type="json_object"),
            extra_body=extra_body,
        )

        if isinstance(model_output, Stream):

            def streamer() -> typing.Generator[GeneratedAnswer, None, None]:
                generated_output = ""
                generated_obj = GeneratedAnswer(sources=[])
                for chunk in model_output:
                    chunk_str = chunk.choices[0].delta.content
                    if chunk_str is None:
                        continue
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

        for suffix in ["", "}", '"}']:
            try:
                generated_dict = json.loads(generated_output + suffix)
                break
            except json.JSONDecodeError:
                continue
        else:
            logger.error(f"Could not decode JSON from model output: {generated_output}")
            return GeneratedAnswer(sources=[], answer="Not JSON-decodable.")

        try:
            generated_obj = GeneratedAnswer.model_validate(generated_dict)
        except ValidationError:
            logger.error(f"Could not validate model output: {generated_dict}")
            return GeneratedAnswer(sources=[], answer="JSON not valid.")

        logger.info(f"Generated answer: {generated_obj.answer!r}")
        return generated_obj


class VllmGenerator(OpenaiGenerator):
    """A generator that uses a vLLM model to generate answers."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the vLLM generator.

        Args:
            config:
                The Hydra configuration.
        """
        logging.getLogger("transformers").setLevel(logging.CRITICAL)

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.generator.model)

        # If an inference server isn't already running then start a new server in a
        # background process and store the process ID
        self.server_process: subprocess.Popen | None
        if config.generator.server is None:
            # We can only run the inference server if CUDA is available
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "The `vLLMGenerator` requires a CUDA-compatible GPU to run. "
                    "Please ensure that a compatible GPU is available and try again."
                )

            config.generator.server = "0.0.0.0"
            self.server_process = self.start_inference_server()
        else:
            self.server_process = None

        super().__init__(config=config)

    def prompt_too_long(self, prompt: str) -> bool:
        """Check if a prompt is too long for the generator.

        Args:
            prompt:
                The prompt to check.

        Returns:
            Whether the prompt is too long for the generator.
        """
        num_tokens = len(self.tokenizer.encode(prompt))
        return num_tokens > self.config.generator.max_input_tokens

    def start_inference_server(self) -> subprocess.Popen:
        """Start the vLLM inference server.

        Returns:
            The inference server process.
        """
        logger.info("Starting vLLM server...")

        # Start server using the vLLM entrypoint
        max_model_len = str(
            self.config.generator.max_input_tokens
            + self.config.generator.max_output_tokens
            + 100
        )
        process = subprocess.Popen(
            args=[
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.config.generator.model,
                "--max-model-len",
                max_model_len,
                "--gpu-memory-utilization",
                str(self.config.generator.gpu_memory_utilization),
                "--chat-template",
                self.tokenizer.chat_template,
                "--host",
                self.config.generator.server,
                "--port",
                str(self.config.generator.port),
                "--swap-space",
                "0",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Wait for the server to start
        stderr = process.stderr
        assert stderr is not None
        for seconds in range(self.config.generator.timeout):
            update = stderr.readline().decode("utf-8")
            if "Uvicorn running" in update:
                logger.info(f"vLLM server started after {seconds} seconds.")
                break
            sleep(1)
        else:
            process.kill()
            raise RuntimeError("vLLM server failed to start.")

        return process

    def __del__(self) -> None:
        """Close down the vLLM server, if we started a new one."""
        if hasattr(self, "server_process") and self.server_process is not None:
            self.server_process.kill()
        del self
