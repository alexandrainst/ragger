"""A Gradio demo of the RAG system."""

import json
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
        self.retrieved_documents: list[Document] = []
        if self.config.demo.mode not in ["strict_feedback", "feedback", "no_feedback"]:
            raise ValueError(
                "The feedback mode must be one of 'strict_feedback'"
                ", 'feedback', or 'no_feedback'."
            )
        if self.config.demo.mode in ["strict_feedback", "feedback"]:
            self.db_path = Path(config.dirs.data) / config.demo.db_path
            self.connection = sqlite3.connect(self.db_path)
            if not self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
            ).fetchone():
                self.connection.execute(
                    (
                        "CREATE TABLE feedback (query text, response text,"
                        "liked integer, document_ids text)"
                    )
                )
                self.connection.commit()
            self.connection.close()

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
                f"<b><center>{self.config.demo.description}</b></center>", label="p"
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
            submit_button_has_added_text_and_asked = submit_button.click(
                fn=self.add_text,
                inputs=[chatbot, input_box, submit_button],
                outputs=[chatbot, input_box, submit_button],
                queue=False,
            ).then(fn=self.ask, inputs=chatbot, outputs=chatbot)
            input_box_has_added_text_and_asked = input_box.submit(
                fn=self.add_text,
                inputs=[chatbot, input_box, submit_button],
                outputs=[chatbot, input_box, submit_button],
                queue=False,
            ).then(fn=self.ask, inputs=chatbot, outputs=chatbot)

            if self.config.demo.mode in ["strict_feedback", "feedback"]:
                submit_button_has_added_text_and_asked.then(
                    fn=lambda: gr.update(
                        value=f"<b><center>{self.config.demo.feedback}</center></b>"
                    ),
                    outputs=[directions],
                    queue=False,
                )

                input_box_has_added_text_and_asked.then(
                    fn=lambda: gr.update(
                        value=f"<b><center>{self.config.demo.feedback}</center></b>"
                    ),
                    outputs=[directions],
                    queue=False,
                )
                chatbot.like(fn=self.vote, inputs=chatbot).then(
                    fn=lambda: (
                        gr.update(interactive=True, visible=True),
                        gr.update(interactive=True, visible=True),
                    ),
                    outputs=[input_box, submit_button],
                    queue=False,
                ).then(
                    fn=lambda: gr.update(
                        value=(
                            "<b><center>"
                            f"{self.config.demo.thank_you_feedback}"
                            "</center></b>"
                        )
                    ),
                    outputs=[directions],
                    queue=False,
                )

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

    def add_text(
        self, history: History, input_text: str, button_text: str
    ) -> tuple[History, dict, dict]:
        """Add the text to the chat history.

        Args:
            history:
                The chat history.
            input_text:
                The text to add.
            button_text:
                The value of the submit button. This is how gradio Button works, when
                used as input to a function.

        Returns:
            The updated chat history, the textbox and updated submit button.
        """
        history = history + [(input_text, None)]
        if self.config.demo.mode == "strict_feedback":
            return (
                history,
                gr.update(value="", interactive=False, visible=False),
                gr.update(value=button_text, interactive=False, visible=False),
            )

        return history, gr.update(value=""), gr.update(value=button_text)

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
        self.retrieved_documents = documents
        history[-1] = (None, generated_answer)
        yield history

    def vote(self, data: gr.LikeData, history: History):
        """Record the vote in the database.

        Args:
            data: The like data.
            history: The chat history.
        """
        retrieved_document_data = {}
        retrieved_document_data["id"] = json.dumps(
            [getattr(document, "id") for document in self.retrieved_documents]
        )
        record = {
            "query": history[-2][0],
            "response": history[-1][1],
            "liked": int(data.liked),
        } | retrieved_document_data

        # Add the record to the table "feedback" in the database.
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute(
            ("INSERT INTO feedback VALUES " "(:query, :response, :liked, :id)"), record
        )
        self.connection.commit()
        self.connection.close()
