"""A Gradio demo of the RAG system."""

import gradio as gr
from omegaconf import DictConfig

from .rag_system import RagSystem


class Demo:
    """A Gradio demo of the RAG system."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the demo.

        Args:
            config:
                The Hydra configuration.
        """
        self.config = config
        self.rag_system = RagSystem(config=config)

    def launch(self) -> None:
        """Launch the demo."""

        def generate(query: str) -> str:
            """Generate an answer to a query.

            Args:
                query:
                    The query to generate an answer to.

            Returns:
                The generated answer.
            """
            answer, documents = self.rag_system.answer(query)
            sources_quote = "'".join(document.text for document in documents)
            sources_ids_str = ", ".join(document.id for document in documents)
            return (
                f"Model answer:\n\t {answer},\n\n"
                f"Based on the following text:\n\t '{sources_quote}',\n\n"
                f"Source for the text:\n\t {sources_ids_str}"
            )

        gr.Interface(
            fn=generate,
            inputs=gr.Textbox(lines=2, label="Query"),
            outputs=gr.Textbox(label="Answer"),
            title="RAG System",
            description="A demo of the RAG system.",
        ).launch()
