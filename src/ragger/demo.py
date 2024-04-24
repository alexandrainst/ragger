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
            answer, documents = self.rag_system.answer(query=query)
            doc_str = "tekster" if len(documents) > 1 else "tekst"
            answer += f"\n\nDette er baseret på følgende {doc_str}:\n\n"
            answer += "\n\n".join(
                f'=== {document.id} ===\n"{document.text}"' for document in documents
            )
            return answer

        self.demo = gr.Interface(
            fn=generate,
            inputs=gr.Textbox(lines=2, label="Spørgsmål"),
            outputs=gr.Textbox(label="Svar"),
            title="RAG System",
            description="En demo af et RAG system.",
            api_name="RAG System",
            allow_flagging="never",
        )
        auth = (
            (self.config.demo.username, self.config.demo.password)
            if self.config.demo.password_protected
            else None
        )
        self.demo.launch(
            server_name=self.config.demo.host,
            server_port=self.config.demo.port,
            share=self.config.demo.share,
            auth=auth,
        )

    def close(self) -> None:
        """Close the demo."""
        if self.demo:
            self.demo.close()
