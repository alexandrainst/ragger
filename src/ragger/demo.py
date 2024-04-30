"""A Gradio demo of the RAG system."""

import sqlite3
import typing
from pathlib import Path

import gradio as gr
from omegaconf import DictConfig

from ragger.utils import Document, format_answer

from .rag_system import RagSystem

Message = str | None
Exchange = tuple[Message, Message]
History = list[Exchange]


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
        if self.config.demo.feedback_mode == "strict_feedback":
            self.db_path = Path(config.dirs.data) / config.demo.db_path
            self.connection = sqlite3.connect(self.db_path)
            if not self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
            ).fetchone():
                self.connection.execute(
                    (
                        "CREATE TABLE feedback (query text, response text,"
                        "liked boolean, sources text)"
                    )
                )
                self.connection.commit()
            self.connection.close()
        elif self.config.demo.feedback_mode == "feedback":
            raise NotImplementedError(
                "Non forced feedback mode 'feedback' is not yet implemented."
            )

    def build_demo(self) -> gr.Blocks:
        """Build the demo.

        Returns:
            The demo.
        """
        with gr.Blocks(
            theme=self.config.demo.theme, title=self.config.demo.title
        ) as demo:
            gr.components.HTML(f"<center><h1>{self.config.demo.title}</h1></center>")
            directions = gr.components.HTML(
                f"<center>{self.config.demo.description}</center>", label="p"
            )
            chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False)
            with gr.Row():
                input_box = gr.Textbox(
                    scale=4,
                    show_label=False,
                    placeholder=self.config.demo.input_box_placeholder,
                    container=False,
                )
            submit_button = gr.Button(
                value=self.config.demo.submit_button_value, variant="primary"
            )
            submit_button.click(
                fn=self.add_text,
                inputs=[chatbot, input_box],
                outputs=[chatbot, input_box],
                queue=False,
            ).then(fn=self.ask, inputs=chatbot, outputs=chatbot).then(
                fn=lambda: gr.update(
                    value=f"<center>{self.config.demo.feedback}</center>"
                ),
                inputs=None,
                outputs=[directions],
                queue=False,
            )

            input_box.submit(
                fn=self.add_text,
                inputs=[chatbot, input_box],
                outputs=[chatbot, input_box],
                queue=False,
            ).then(fn=self.ask, inputs=chatbot, outputs=chatbot)
        return demo

    def launch(self) -> None:
        """Launch the demo."""
        self.demo = self.build_demo()
        auth = (
            (self.config.demo.username, self.config.demo.password)
            if self.config.demo.password_protected
            else None
        )
        self.demo.queue().launch(
            server_name=self.config.demo.host,
            server_port=self.config.demo.port,
            share=self.config.demo.share,
            auth=auth,
        )

    def close(self) -> None:
        """Close the demo."""
        if self.demo:
            self.demo.close()

    @staticmethod
    def add_text(history: History, text: str) -> tuple[History, gr.Textbox]:
        """Add the text to the chat history.

        Args:
            history:
                The chat history.
            text:
                The text to add.

        Returns:
            The updated chat history and the updated chatbot.
        """
        history = history + [(text, None)]
        return history, gr.Textbox(value="")

    def ask(self, history: History) -> typing.Generator[History, None, None]:
        """Ask the bot a question.

        Args:
            history:
                The chat history.

        Returns:
            The updated chat history.
        """
        human_message: str = history[-1][0] if history[-1][0] else ""
        empty_exhange: Exchange = (None, "")
        history.append(empty_exhange)
        answer_or_stream = self.rag_system.answer(query=human_message)
        if isinstance(answer_or_stream, typing.Generator):
            generated_answer = ""
            documents: list[Document] = []
            for generated_answer, documents in answer_or_stream:
                assert isinstance(generated_answer, str)
                history[-1] = (None, generated_answer)
                yield history
        else:
            generated_answer, documents = answer_or_stream
        generated_answer = format_answer(
            answer=generated_answer,
            documents=documents,
            no_documents_reply=self.config.demo.no_documents_reply,
        )
        history[-1] = (None, generated_answer)
        yield history
