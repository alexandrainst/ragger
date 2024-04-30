"""Script to run a local CLI demo, mostly used for testing.

Usage:
    python src/scripts/run_cli.py <key>=<value> <key>=<value> ...
"""

import typing

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from ragger.rag_system import RagSystem
from ragger.utils import Document, format_answer

load_dotenv()


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Run the CLI.

    Args:
        config:
            The Hydra configuration.
    """
    rag_system = RagSystem(config=config)
    while True:
        query = input(f"{config.demo.input_box_placeholder.strip()}: ")
        answer_or_stream = rag_system.answer(query=query)
        if isinstance(answer_or_stream, typing.Generator):
            generated_answer = ""
            documents: list[Document] = []
            for generated_answer, documents in answer_or_stream:
                print(generated_answer, end="\r")
        else:
            generated_answer, documents = answer_or_stream
        answer = format_answer(
            answer=generated_answer,
            documents=documents,
            no_documents_reply=config.demo.no_documents_reply,
        )
        print(answer)
        print()


if __name__ == "__main__":
    main()
