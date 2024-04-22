"""GUI to interact with the QA bot."""

import logging
import warnings
from pathlib import Path
from typing import Any, Generator

import gradio as gr
import yaml
from omegaconf import DictConfig

from .qa_bot import QABot
from .types import Exchange, History

logger = logging.getLogger(__name__)


def build_gui(cfg: DictConfig) -> gr.Blocks:
    """Builds a gui for the bot.

    Args:
        cfg:
            The Hydra configuration object.

    Returns:
        The gui.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    qa_bot = QABot(cfg=cfg)

    def ask(history: History) -> Generator[History, None, None]:
        """Ask the bot a question.

        Args:
            history:
                The chat history.

        Yields:
            The chat history.
        """
        human_message: str = history[-1][0] if history[-1][0] else ""
        logger.info(f"Asked the following question: {human_message!r}")

        # Add a placeholder for the bot's response
        last_bot_message = ""
        empty_exhange: Exchange = (None, last_bot_message)
        history.append(empty_exhange)

        # Answer the question and yield the answer
        answer = qa_bot(history=history)
        if isinstance(answer, str):
            history[-1] = (None, answer)
            logger.info(f"Generated the following answer: {answer!r}")
            yield history
        else:
            generated_answer: str = ""
            for chunk in answer:
                generated_answer += chunk
                history[-1] = (None, generated_answer)
                yield history
            logger.info(f"Generated the following answer: {generated_answer!r}")

    def add_text(history: History, text: str) -> tuple[History, dict[Any, Any]]:
        """Add text to the chat history.

        Args:
            history:
                The chat history.
            text:
                The text to add to the chat history.

        Returns:
            The chat history and the update dictionary.
        """
        history = history + [(text, None)]
        return history, gr.update(value="", interactive=False)

    def vote(data: gr.LikeData, history: History) -> None:
        """Save the user's vote.

        Args:
            data:
                The like data.
            history:
                The chat history.
        """
        answer_store_dir = (
            Path(cfg.dirs.data) / cfg.dirs.processed / cfg.poc1.gui.answer_store
        )
        record = {
            "query": history[-2][0],
            "response": history[-1][1],
            "liked": data.liked,
        }
        logger.debug(f"Saving the following record: {record}")

        # Get the answer store and append the new record.
        if not answer_store_dir.exists():
            answer_store_dir.mkdir(parents=True)
            current_answer_store = []
        else:
            with answer_store_dir.open("r") as f:
                current_answer_store = yaml.safe_load(f)
                current_answer_store.append(record)
        with answer_store_dir.open("w") as f:
            yaml.safe_dump(current_answer_store, f)

    title = "<center><h1>QA Bot</h1></center>"
    description = "<center>Hej! Stil mig spørgsmål nedenfor.</center>"
    with gr.Blocks(title="QA Bot", theme=gr.themes.Soft()) as gui:
        gr.components.HTML(title)
        gr.components.HTML(description, label="p")
        chatbot = gr.Chatbot(value=[], elem_id="chatbot", bubble_full_width=False)

        with gr.Row():
            input_box = gr.Textbox(
                scale=4,
                show_label=False,
                placeholder='Stil dit spørgsmål og tryk på "Send".',
                container=False,
            )

        # Submit button is clicked, add the text to the history and ask the bot.
        submit_button = gr.Button(value="Send", variant="primary")
        submit_button.click(
            fn=add_text,
            inputs=[chatbot, input_box],
            outputs=[chatbot, input_box],
            queue=False,
        ).then(fn=ask, inputs=chatbot, outputs=chatbot).then(
            fn=lambda: gr.update(interactive=True),
            inputs=None,
            outputs=[input_box],
            queue=False,
        )

        # Enter key is pressed, add the text to the history and ask the bot.
        input_box.submit(
            fn=add_text,
            inputs=[chatbot, input_box],
            outputs=[chatbot, input_box],
            queue=False,
        ).then(fn=ask, inputs=chatbot, outputs=chatbot).then(
            fn=lambda: gr.update(interactive=True),
            inputs=None,
            outputs=[input_box],
            queue=False,
        )

        # Like button is clicked, upvote the bot's response.
        chatbot.like(fn=vote, inputs=chatbot, outputs=None)
    return gui.queue()
