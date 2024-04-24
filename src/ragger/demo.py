"""A Gradio demo of the RAG system."""

from typing import Generator

import gradio as gr
from omegaconf import DictConfig

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

    def build_demo(self) -> gr.Blocks:
        """Build the demo.

        Returns:
            The demo.
        """
        with gr.Blocks(
            theme=self.config.demo.theme, title=self.config.demo.title
        ) as demo:
            gr.components.HTML(f"<center><h1>{self.config.demo.title}</h1></center>")
            gr.components.HTML(
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
                self.add_text, [chatbot, input_box], [chatbot, input_box], queue=False
            ).then(self.ask, chatbot, chatbot)
            input_box.submit(
                self.add_text, [chatbot, input_box], [chatbot, input_box], queue=False
            ).then(self.ask, chatbot, chatbot)
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

    def ask(self, history: History) -> Generator[History, None, None]:
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
        answer, documents = self.rag_system.answer(query=human_message)
        match len(documents):
            case 0:
                answer = self.config.demo.no_documents_reply
            case 1:
                answer += "\n\nKilde:\n\n"
            case _:
                answer += "\n\nKilder:\n\n"
        answer += "\n\n".join(
            f'===  {document.id}  ===\n"{document.text}"' for document in documents
        )
        history[-1] = (None, answer)
        yield history
