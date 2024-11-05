"""Generation of an answer from a query and a list of relevant documents."""

import json
import logging
import os
import subprocess
import typing
from functools import cached_property
from time import sleep

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray
from pydantic import ValidationError
from pydantic_core import from_json

from .constants import (
    DANISH_SYSTEM_PROMPT,
    DANISH_USER_PROMPT,
    ENGLISH_SYSTEM_PROMPT,
    ENGLISH_USER_PROMPT,
)
from .data_models import Document, GeneratedAnswer, Generator, Retriever
from .utils import is_installed, raise_if_not_installed

if is_installed(package_name="torch"):
    import torch

if is_installed(package_name="httpx"):
    from httpx import ReadTimeout, RemoteProtocolError

if is_installed(package_name="openai"):
    from openai import (
        APITimeoutError,
        InternalServerError,
        LengthFinishReasonError,
        OpenAI,
    )
    from openai.lib.streaming.chat import (
        ChatCompletionStreamManager,
        ChunkEvent,
        ContentDeltaEvent,
        ContentDoneEvent,
    )
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
        ParsedChatCompletion,
    )

if is_installed(package_name="tiktoken"):
    import tiktoken

if is_installed(package_name="transformers"):
    from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

if is_installed(package_name="llama_cpp"):
    from llama_cpp import (
        ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessage,
        Llama,
        LogitsProcessor,
        LogitsProcessorList,
    )

if is_installed(package_name="outlines"):
    from outlines.models.transformers import TransformerTokenizer
    from outlines.processors.structured import JSONLogitsProcessor

if is_installed(package_name="huggingface_hub"):
    from huggingface_hub import HfApi

if typing.TYPE_CHECKING:
    import tiktoken
    from httpx import ReadTimeout, RemoteProtocolError
    from huggingface_hub import HfApi
    from llama_cpp import (
        ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessage,
        Llama,
        LogitsProcessor,
        LogitsProcessorList,
    )
    from openai import (
        APITimeoutError,
        InternalServerError,
        LengthFinishReasonError,
        OpenAI,
    )
    from openai.lib.streaming.chat import (
        ChatCompletionStreamManager,
        ChunkEvent,
        ContentDeltaEvent,
        ContentDoneEvent,
    )
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
        ParsedChatCompletion,
    )
    from outlines.models.transformers import TransformerTokenizer
    from outlines.processors.structured import JSONLogitsProcessor
    from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

    from .data_models import DocumentStore


load_dotenv()


logger = logging.getLogger(__package__)


class OpenAIGenerator(Generator):
    """A generator that uses an OpenAI model to generate answers."""

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key: str | None = None,
        host: str | None = None,
        port: int = 8000,
        timeout: int = 60,
        max_retries: int = 3,
        max_input_tokens: int = 130_000,
        max_output_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = False,
        language: typing.Literal["da", "en"] = "da",
        system_prompt: str | None = None,
        prompt: str | None = None,
        **additional_generation_kwargs,
    ) -> None:
        """Initialise the OpenAI generator.

        Args:
            model_id (optional):
                The OpenAI model ID. Defaults to "gpt-4o-mini".
            api_key (optional):
                The OpenAI API key, or None if it should be read from the environment
                variable "OPENAI_API_KEY", or if it is simply not needed (e.g., if
                `host` is provided).
            host (optional):
                The host of the OpenAI server, if different from the default.
            port (optional):
                The port of the OpenAI server. Defaults to 8000.
            timeout (optional):
                The timeout for the OpenAI requests, in seconds. Defaults to 60.
            max_retries (optional):
                The maximum number of retries for the OpenAI requests. Defaults
                to 3.
            max_input_tokens (optional):
                The maximum number of tokens allowed in the input. Defaults to
                130,000.
            max_output_tokens (optional):
                The maximum number of tokens allowed in the output. Defaults to
                256.
            temperature (optional):
                The temperature of the model. Defaults to 0.0.
            stream (optional):
                Whether to stream the output. Defaults to False.
            language (optional):
                The language of the model. Can be "da" (Danish) or "en" (English).
                Defaults to "da".
            system_prompt (optional):
                The system prompt to use. If None, the default system prompt
                corresponding to the chosen language will be used.
            prompt (optional):
                The prompt to use. If None, the default prompt corresponding to
                the chosen language will be used.
            additional_generation_kwargs (optional):
                Additional keyword arguments to pass to the generation function.
        """
        raise_if_not_installed(
            package_names=["openai", "tiktoken", "httpx"], extra="openai"
        )

        self.model_id = model_id
        self.api_key = api_key
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.stream = stream
        self.language = language
        self.additional_generation_kwargs = additional_generation_kwargs

        # Set the system and user prompts based on the language
        system_prompt_mapping = dict(da=DANISH_SYSTEM_PROMPT, en=ENGLISH_SYSTEM_PROMPT)
        user_prompt_mapping = dict(da=DANISH_USER_PROMPT, en=ENGLISH_USER_PROMPT)
        self.system_prompt = system_prompt or system_prompt_mapping[self.language]
        self.prompt = prompt or user_prompt_mapping[self.language]

        # Set the server URL, if a host is provided
        self.server: str | None
        if self.host is not None:
            if not self.host.startswith("http"):
                self.host = f"http://{host}"
            self.server = f"{self.host}:{self.port}/v1"
        else:
            self.server = None

        self.client = OpenAI(
            base_url=self.server,
            api_key=api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def prompt_too_long(self, prompt: str) -> bool:
        """Check if a prompt is too long for the generator.

        Args:
            prompt:
                The prompt to check.

        Returns:
            Whether the prompt is too long for the generator.
        """
        encoding = tiktoken.encoding_for_model(model_name=self.model_id)
        num_tokens = len(encoding.encode(text=prompt))
        return num_tokens > self.max_input_tokens

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
            messages: list[ChatCompletionMessageParam] = [
                ChatCompletionSystemMessageParam(
                    role="system", content=self.system_prompt
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=self.prompt.format(
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

            if self.prompt_too_long(prompt=json.dumps(messages)):
                continue

            try:
                model_output: (
                    ChatCompletionStreamManager[GeneratedAnswer] | ParsedChatCompletion
                )
                if self.stream:
                    model_output = self.client.beta.chat.completions.stream(
                        messages=messages,
                        model=self.model_id,
                        max_completion_tokens=self.max_output_tokens,
                        temperature=self.temperature,
                        stop=["</answer>"],
                        response_format=GeneratedAnswer,
                        extra_body=self.additional_generation_kwargs,
                    )
                else:
                    model_output = self.client.beta.chat.completions.parse(
                        messages=messages,
                        model=self.model_id,
                        max_completion_tokens=self.max_output_tokens,
                        temperature=self.temperature,
                        stop=["</answer>"],
                        response_format=GeneratedAnswer,
                        extra_body=self.additional_generation_kwargs,
                    )
            except (InternalServerError, APITimeoutError):
                continue

            # When model output is too long
            except LengthFinishReasonError:
                logger.error(
                    f"Model output too long (>{self.max_output_tokens} tokens) "
                    f"for query: {query}"
                )
                return GeneratedAnswer(sources=[], answer="Not JSON-decodable.")

            # If we are streaming we try to get a sample from the stream to check if
            # the prompt is too long, as we cannot check for this in advance
            if isinstance(model_output, ChatCompletionStreamManager):
                try:
                    with model_output as stream:
                        next(stream)
                except (RemoteProtocolError, ReadTimeout):
                    continue

            break
        else:
            return GeneratedAnswer(sources=[], answer="Prompt too long.")

        if isinstance(model_output, ChatCompletionStreamManager):

            def streamer() -> typing.Generator[GeneratedAnswer, None, None]:
                generated_output = ""
                generated_obj = GeneratedAnswer(answer="", sources=[])
                with model_output as stream:
                    for chunk_event in stream:
                        if isinstance(chunk_event, ContentDeltaEvent):
                            chunk_str = chunk_event.delta
                            generated_output += chunk_str
                        elif isinstance(chunk_event, ChunkEvent):
                            chunk_str_or_none = chunk_event.chunk.choices[
                                0
                            ].delta.content
                            if chunk_str_or_none is None:
                                continue
                            generated_output += chunk_str_or_none
                        elif isinstance(chunk_event, ContentDoneEvent):
                            generated_output = chunk_event.content
                        else:
                            logger.error(
                                "Unknown event type received during generation: "
                                f"{chunk_event}. Skipping."
                            )
                            continue
                        try:
                            generated_dict = from_json(
                                data=generated_output, allow_partial=True
                            )

                            # If the sources in the generated JSON dict is empty, but
                            # the final closing square bracket hasn't been generated
                            # yet, this means that the `from_json` function has closed
                            # this off itself, which is not allowed here, as this would
                            # trigger the "cannot answer" answer. To prevent this, we
                            # check for this and skip the next chunk if this is the
                            # case.
                            first_source_not_generated_yet = (
                                "sources" in generated_dict
                                and not generated_dict["sources"]
                                and '"sources": []' not in generated_output
                            )
                            if first_source_not_generated_yet:
                                continue

                            # If the answer is being written, the JSON dict will look
                            # like
                            #   '{"sources": [...], "answer": "Some text'
                            # As the answer doesn't have a closing quote, the
                            # `from_json` function will not include the `answer` key in
                            # the resulting dict. To ensure that the partial answer *is*
                            # included in the dict, we check if the model is currently
                            # writing the answer and if so, we add a closing quote to
                            # the generated output before attempting to parse it.
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
                            generated_obj = GeneratedAnswer.model_validate(
                                generated_dict
                            )
                            yield generated_obj
                        except ValidationError:
                            continue

            return streamer()
        else:
            generated_output = model_output.choices[0].message.content
            assert generated_output is not None
            generated_output = generated_output.strip()

        for suffix in ["", "}", '"}']:
            try:
                generated_dict = json.loads(generated_output + suffix)
                break
            except json.JSONDecodeError:
                continue
        else:
            logger.error(f"Could not decode JSON from model output: {generated_output}")
            return GeneratedAnswer(sources=[], answer="Not JSON-decodable.")

        generated_obj = GeneratedAnswer.model_validate(generated_dict)
        logger.info(f"Generated answer: {generated_obj.answer!r}")
        return generated_obj


class VllmGenerator(OpenAIGenerator):
    """A generator that uses a vLLM model to generate answers."""

    def __init__(
        self,
        model_id: str = "AI-Sweden-Models/Llama-3-8B-instruct",
        host: str | None = None,
        port: int = 8000,
        timeout: int = 60,
        max_retries: int = 3,
        max_input_tokens: int = 10_000,
        max_output_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = True,
        language: typing.Literal["da", "en"] = "da",
        system_prompt: str | None = None,
        prompt: str | None = None,
        gpu_memory_utilization: float = 0.95,
        server_start_timeout: int = 60,
    ) -> None:
        """Initialise the vLLM generator.

        Args:
            model_id (optional):
                The model ID of the generative model to use. Defaults to
                "AI-Sweden-Models/Llama-3-8B-instruct".
            host (optional):
                The host of the vLLM server, if it is already running. If None, a new
                server will be started.
            port (optional):
                The port of the vLLM server. Defaults to 8000.
            timeout (optional):
                The timeout for the vLLM requests, in seconds. Defaults to 60.
            max_retries (optional):
                The maximum number of retries for the vLLM requests. Defaults
                to 3.
            max_input_tokens (optional):
                The maximum number of tokens allowed in the input. Defaults to
                10,000.
            max_output_tokens (optional):
                The maximum number of tokens allowed in the output. Defaults to
                256.
            temperature (optional):
                The temperature of the model. Defaults to 0.0.
            stream (optional):
                Whether to stream the output. Defaults to True.
            language (optional):
                The language of the model. Can be "da" (Danish) or "en" (English).
                Defaults to "da".
            system_prompt (optional):
                The system prompt to use. If None, the default system prompt
                corresponding to the chosen language will be used.
            prompt (optional):
                The prompt to use. If None, the default prompt corresponding to
                the chosen language will be used.
            gpu_memory_utilization (optional):
                The fraction of the GPU memory to use. Defaults to 0.95.
            server_start_timeout (optional):
                The timeout for the vLLM server to start, in seconds. Only relevant if
                `host` has been set. Defaults to 60.
        """
        raise_if_not_installed(
            package_names=["vllm", "transformers"], extra="onprem_gpu"
        )

        logging.getLogger("transformers").setLevel(logging.CRITICAL)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=os.getenv("HUGGINGFACE_HUB_TOKEN", True)
        )
        self.hf_config = AutoConfig.from_pretrained(
            model_id, token=os.getenv("HUGGINGFACE_HUB_TOKEN", True)
        )
        self.gpu_memory_utilization = gpu_memory_utilization
        self.server_start_timeout = server_start_timeout

        # If an inference server isn't already running then start a new server in a
        # background process and store the process ID
        self.server_process: subprocess.Popen | None
        if host is None:
            # We can only run the inference server if CUDA is available
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "The `vLLMGenerator` requires a CUDA-compatible GPU to run. "
                    "Please ensure that a compatible GPU is available and try again."
                )
            host = "0.0.0.0"
            self.server_process = self.start_inference_server(host=host, port=port)
        else:
            self.server_process = None

        super().__init__(
            model_id=model_id,
            host=host,
            port=port,
            timeout=timeout,
            max_retries=max_retries,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stream=stream,
            language=language,
            system_prompt=system_prompt,
            prompt=prompt,
            additional_generation_kwargs=dict(
                guided_json=GeneratedAnswer.model_json_schema()
            ),
        )

    @cached_property
    def max_model_length(self) -> int:
        """Get the maximum model length.

        Returns:
            The maximum model length.
        """
        max_model_len_candidates = [
            self.max_input_tokens + self.max_output_tokens,
            10_000 - self.max_output_tokens,  # Upper limit of 10k
        ]
        if (
            hasattr(self.tokenizer, "model_max_length")
            and self.tokenizer.model_max_length
        ):
            max_model_len_candidates.append(
                self.tokenizer.model_max_length - self.max_output_tokens
            )
        if (
            hasattr(self.hf_config, "max_position_embeddings")
            and self.hf_config.max_position_embeddings
        ):
            max_model_len_candidates.append(
                self.hf_config.max_position_embeddings - self.max_output_tokens
            )

        max_model_len = min(max_model_len_candidates)
        logger.info(f"Max model length set to {max_model_len:,} tokens.")
        return max_model_len

    def prompt_too_long(self, prompt: str) -> bool:
        """Check if a prompt is too long for the generator.

        Args:
            prompt:
                The prompt to check.

        Returns:
            Whether the prompt is too long for the generator.
        """
        num_tokens = len(self.tokenizer.encode(prompt))
        return num_tokens > self.max_model_length

    def start_inference_server(self, host: str, port: int) -> subprocess.Popen:
        """Start the vLLM inference server.

        Args:
            host:
                The host to start the server on.
            port:
                The port to start the server on.

        Returns:
            The inference server process.
        """
        logger.info("Loading/downloading model and starting vLLM server...")

        process_args = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--swap-space",
            "0",
            "--enforce-eager",
            "--model",
            self.model_id,
            "--max-model-len",
            str(self.max_model_length),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--host",
            host,
            "--port",
            str(port),
        ]
        if self.tokenizer.chat_template:
            process_args.extend(["--chat-template", self.tokenizer.chat_template])

        process = subprocess.Popen(
            args=process_args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

        # Get the stderr output from the process
        stderr = process.stderr
        assert stderr is not None

        # Wait for the server to start. The `set_blocking` removes blocking from the
        # `readline` method, so that we can check for updates from the server while
        # waiting for it to start.
        os.set_blocking(stderr.fileno(), False)
        error_message = ""
        for seconds in range(self.server_start_timeout):
            update = stderr.readline().decode()
            if not update and error_message:
                process.kill()
                raise RuntimeError(
                    "vLLM server failed to start with the error message "
                    + error_message.strip()
                )
            elif "error" in update.lower() or error_message:
                error_message += update
                continue
            elif "Uvicorn running" in update:
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


class GGUFGenerator(Generator):
    """A generator to generate answers from a model in GGUF format."""

    def __init__(
        self,
        model_id: str = "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        quant_type: str | None = None,
        max_input_tokens: int = 130_000,
        max_output_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = False,
        language: typing.Literal["da", "en"] = "da",
        system_prompt: str | None = None,
        prompt: str | None = None,
        **additional_generation_kwargs,
    ) -> None:
        """Initialise the GGUF generator.

        Args:
            model_id (optional):
                The model ID of the generative model to use. Defaults to
                "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF".
            quant_type (optional):
                The quantization type to use. If None, we will use any of the GGUF
                files available. Defaults to None.
            max_input_tokens (optional):
                The maximum number of tokens allowed in the input. Defaults to
                130,000.
            max_output_tokens (optional):
                The maximum number of tokens allowed in the output. Defaults to
                256.
            temperature (optional):
                The temperature of the model. Defaults to 0.0.
            stream (optional):
                Whether to stream the output. Defaults to False.
            language (optional):
                The language of the model. Can be "da" (Danish) or "en" (English).
                Defaults to "da".
            system_prompt (optional):
                The system prompt to use. If None, the default system prompt
                corresponding to the chosen language will be used.
            prompt (optional):
                The prompt to use. If None, the default prompt corresponding to
                the chosen language will be used.
            additional_generation_kwargs (optional):
                Additional keyword arguments to pass to the generation function.
        """
        raise_if_not_installed(
            package_names=["outlines", "llama_cpp", "transformers"],
            installation_alias_mapping=dict(llama_cpp="llama_cpp_python"),
            extra="onprem_cpu",
        )
        self.model_id = model_id
        self.quant_type = quant_type
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.stream = stream
        self.language = language
        self.additional_generation_kwargs = additional_generation_kwargs

        # Set the system and user prompts based on the language
        system_prompt_mapping = dict(da=DANISH_SYSTEM_PROMPT, en=ENGLISH_SYSTEM_PROMPT)
        user_prompt_mapping = dict(da=DANISH_USER_PROMPT, en=ENGLISH_USER_PROMPT)
        self.system_prompt = system_prompt or system_prompt_mapping[self.language]
        self.prompt = prompt or user_prompt_mapping[self.language]

        self.base_model_id: str | None = None
        self.tokenizer: PreTrainedTokenizer | None = None
        self.logits_processor: LogitsProcessor | None = None
        self.model: Llama | None = None

    def prompt_too_long(self, prompt: str) -> bool:
        """Check if a prompt is too long for the generator.

        Args:
            prompt:
                The prompt to check.

        Returns:
            Whether the prompt is too long for the generator.

        Raises:
            RuntimeError:
                If the generator has not been compiled.
        """
        if self.tokenizer is None:
            raise RuntimeError("The generator has not been compiled.")
        num_tokens = len(self.tokenizer.encode(prompt))
        return num_tokens > self.max_input_tokens

    def compile(self, document_store: "DocumentStore", retriever: "Retriever") -> None:
        """Compile the embedder.

        Args:
            document_store:
                The document store to use.
            retriever:
                The retriever to use.
        """
        self.base_model_id = self._get_base_model_id(model_id=self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, token=os.getenv("HUGGINGFACE_HUB_TOKEN", True)
        )
        assert self.tokenizer is not None
        self.logits_processor = self._get_logits_processor(tokenizer=self.tokenizer)
        self.model = self._load_model(
            model_id=self.model_id, quant_type=self.quant_type
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
        if self.model is None or self.logits_processor is None:
            raise RuntimeError("The generator has not been compiled.")

        for num_documents_to_include in range(len(documents), -1, -1):
            logger.info(
                f"Generating answer for the query {query!r} and "
                f"{num_documents_to_include:,} documents..."
            )

            messages: list[ChatCompletionRequestMessage] = [
                ChatCompletionRequestSystemMessage(
                    role="system", content=self.system_prompt
                ),
                ChatCompletionRequestUserMessage(
                    role="user",
                    content=self.prompt.format(
                        documents=json.dumps(
                            [document.model_dump() for document in documents]
                        ),
                        query=query,
                    ),
                ),
            ]

            if not self.prompt_too_long(prompt=json.dumps(messages)):
                break
        else:
            return GeneratedAnswer(sources=[], answer="Prompt too long.")

        model_output = self.model.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            stream=self.stream,
            logits_processor=LogitsProcessorList([self.logits_processor]),
        )
        if isinstance(model_output, dict):
            generated_output = model_output["choices"][0]["message"]["content"]
            if generated_output is None:
                return GeneratedAnswer(sources=[], answer="No answer generated.")
            generated_output = generated_output.strip()

            try:
                generated_dict = json.loads(generated_output)
            except json.JSONDecodeError:
                logger.error(
                    f"Could not decode JSON from model output: {generated_output}"
                )
                return GeneratedAnswer(sources=[], answer="Not JSON-decodable.")

            generated_obj = GeneratedAnswer.model_validate(generated_dict)
            logger.info(f"Generated answer: {generated_obj.answer!r}")
            return generated_obj

        else:

            def streamer() -> typing.Generator[GeneratedAnswer, None, None]:
                generated_output = ""
                generated_obj = GeneratedAnswer(answer="", sources=[])
                for chunk in model_output:
                    delta = chunk["choices"][0]["delta"]
                    if "content" not in delta:
                        continue
                    delta_content = delta["content"]  # type: ignore[typeddict-item]
                    if delta_content is None:
                        continue
                    generated_output += delta_content

                    try:
                        generated_dict = from_json(
                            data=generated_output, allow_partial=True
                        )

                        # If the sources in the generated JSON dict is empty, but the
                        # final closing square bracket hasn't been generated yet, this
                        # means that the `from_json` function has closed this off
                        # itself, which is not allowed here, as this would trigger the
                        # "cannot answer" answer. To prevent this, we check for this and
                        # skip the next chunk if this is the case.
                        first_source_not_generated_yet = (
                            "sources" in generated_dict
                            and not generated_dict["sources"]
                            and '"sources": []' not in generated_output
                        )
                        if first_source_not_generated_yet:
                            continue

                        # If the answer is being written, the JSON dict will look like
                        # '{"sources": [...], "answer": "Some text' As the answer
                        # doesn't have a closing quote, the `from_json` function will
                        # not include the `answer` key in the resulting dict. To ensure
                        # that the partial answer *is* included in the dict, we check if
                        # the model is currently writing the answer and if so, we add a
                        # closing quote to the generated output before attempting to
                        # parse it.
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

    @staticmethod
    def _get_base_model_id(model_id: str) -> str:
        """Get the base model ID.

        Returns:
            The base model ID.

        Raises:
            ValueError:
                If the model ID is not in the correct "author/model_name" format.
        """
        if model_id.count("/") != 1:
            raise ValueError(
                f"Model ID {model_id!r} is not in the correct format. "
                "It should be in the format 'author/model_name'."
            )
        author, model_name = model_id.split("/")

        api = HfApi()
        models = [
            model
            for model in api.list_models(author=author, model_name=model_name)
            if model.id == model_id
        ]

        # Check that the model exists. If it does not then raise an error
        if len(models) == 0:
            logger.error(
                f"Could not find model with ID {model_id} on the Hugging Face "
                "Hub. Using the model ID as the base model ID."
            )
            return model_id

        base_model_tag_candidates = [
            tag
            for tag in models[0].tags or list()
            if tag.startswith("base_model:") and tag.count(":") == 1
        ]
        if not base_model_tag_candidates:
            logger.error(
                f"Could not find a base model tag for model with ID {model_id}. "
                "Using the model ID as the base model ID."
            )
            return model_id

        return base_model_tag_candidates[0].split(":")[1]

    @staticmethod
    def _get_logits_processor(tokenizer: "PreTrainedTokenizer") -> "LogitsProcessor":
        """Get the logits processor.

        Args:
            tokenizer:
                The tokenizer to use.

        Returns:
            The logits processor.
        """
        logits_processor = JSONLogitsProcessor(
            schema=GeneratedAnswer, tokenizer=TransformerTokenizer(tokenizer)
        )

        def logits_processor_fn(
            input_ids: NDArray[np.intc], logits: NDArray[np.single]
        ) -> NDArray[np.single]:
            raw_output = logits_processor(input_ids=input_ids, logits=logits)
            assert isinstance(
                raw_output, np.ndarray
            ), f"Logits processor returned an invalid type: {type(raw_output)}"
            return raw_output

        return logits_processor_fn

    @staticmethod
    def _load_model(model_id: str, quant_type: str | None) -> "Llama":
        """Load the model.

        Args:
            model_id:
                The model ID to use.
            quant_type:
                The quantization type to use.

        Returns:
            The model.

        Raises:
            ValueError:
                If no model with the given quantization type could be found.
        """
        try:
            glob_pattern = "".join(
                f"[{char.upper()}{char.lower()}]" if char.isalpha() else char
                for char in quant_type or ""
            )
            logger.info(f"Loading model with quantization type {quant_type!r}...")
            return Llama.from_pretrained(
                repo_id=model_id, filename=f"*{glob_pattern}.gguf"
            )
        except ValueError as e:
            if "Multiple files found" not in str(e):
                raise e
            for num_bits in range(8, 0, -1):
                try:
                    logger.info(f"Loading model with quantization type Q{num_bits}*...")
                    return Llama.from_pretrained(
                        repo_id=model_id, filename=f"*[Qq]{num_bits}*.gguf"
                    )
                except ValueError:
                    logger.error(
                        f"Could not find model with quantization type Q{num_bits}. "
                        "Trying with a lower quantization type..."
                    )
                    continue
            raise ValueError(
                f"Could not find any quantized model for model ID {model_id}."
            )
