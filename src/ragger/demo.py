"""A Gradio demo of the RAG system."""

import json
import logging
import os
import sqlite3
import typing
from pathlib import Path

import gradio as gr
import huggingface_hub
from huggingface_hub import CommitScheduler, HfApi, create_repo
from huggingface_hub.utils import LocalTokenNotFoundError
from omegaconf import DictConfig

from .rag_system import RagSystem
from .utils import Document, format_answer

Message = str | None
Exchange = tuple[Message, Message]
History = list[Exchange]


logger = logging.getLogger(__package__)


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

        # Initialise commit scheduler, which will commit the final data directory to
        # the Hub at regular intervals. This is only necessary when running in a
        # Hugging Face Space, which we control via the `RUNNING_IN_SPACE` environment
        # variable.
        if os.getenv("RUNNING_IN_SPACE") == "1":
            final_data_path = Path(self.config.dirs.data) / self.config.dirs.final
            assert final_data_path.exists(), f"{final_data_path!r} does not exist!"
            self.scheduler = CommitScheduler(
                repo_id=self.config.demo.persistent_sharing.repo_id,
                repo_type="space",
                folder_path=final_data_path,
                path_in_repo=str(final_data_path),
                squash_history=True,
                every=5,
            )

    def build_demo(self) -> gr.Blocks:
        """Build the demo.

        Returns:
            The demo.
        """
        logger.info("Building the demo...")
        with gr.Blocks(
            theme=self.config.demo.theme, title=self.config.demo.title, fill_height=True
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
        logger.info("Built the demo.")
        return demo

    def launch(self) -> None:
        """Launch the demo."""
        self.demo = self.build_demo()
        logger.info("Launching the demo...")
        launch_kwargs = dict(
            server_name=self.config.demo.host, server_port=self.config.demo.port
        )
        match self.config.demo.share:
            case "temporary":
                auth = None
                username = self.config.demo.temporary_sharing.username
                password = self.config.demo.temporary_sharing.password
                if (
                    isinstance(username, str)
                    and username != ""
                    and isinstance(password, str)
                    and password != ""
                ):
                    auth = (username, password)
                launch_kwargs |= dict(share=True, auth=auth)
                self.demo.queue().launch(**launch_kwargs)
            case "persistent":
                self.push_to_hub()
            case "no-share":
                self.demo.queue().launch(**launch_kwargs)
            case _:
                raise ValueError(
                    "The `demo.share` field in the config must be one of 'temporary', "
                    "'persistent', or 'no-share'. It is currently set to "
                    f"{self.config.demo.share!r}. Please change it and try again."
                )

    def push_to_hub(self) -> None:
        """Pushes the demo to a Hugging Face Space on the Hugging Face Hub."""
        if self.config.demo.share != "persistent":
            raise ValueError(
                "The demo must be shared persistently to push it to the hub. Please "
                "set the `demo.share` field to 'persistent' in the config and try "
                "again."
            )
        if self.config.demo.persistent_sharing.repo_id is None:
            raise ValueError(
                "The `demo.persistent_sharing.repo_id` field must be set in the "
                "config to push the demo to the hub. Please set it and try again."
            )

        # Check that all environment variables are set
        required_env_vars: list[str] = []
        try:
            huggingface_hub.whoami()
        except LocalTokenNotFoundError:
            required_env_vars.append(
                self.config.demo.persistent_sharing.token_variable_name
            )
        if self.config.generator.name == "openai":
            required_env_vars.append(self.config.generator.api_key_variable_name)
        for env_var in required_env_vars:
            if env_var not in os.environ:
                raise ValueError(
                    f"{env_var} environment variable is not set. Please set it in "
                    "your `.env` file or in your environment, and try again."
                )

        logger.info("Pushing the demo to the hub...")

        api = HfApi(
            token=os.environ[self.config.demo.persistent_sharing.token_variable_name]
        )

        if not api.repo_exists(repo_id=self.config.demo.persistent_sharing.repo_id):
            create_repo(
                repo_id=self.config.demo.persistent_sharing.repo_id,
                repo_type="space",
                space_sdk="docker",
                exist_ok=True,
                private=True,
            )

        # This environment variable is used to trigger the creation of a commit
        # scheduler when the demo is initialised, which will commit the final data
        # directory to the Hub at regular intervals.
        api.add_space_variable(
            repo_id=self.config.demo.persistent_sharing.repo_id,
            key="RUNNING_IN_SPACE",
            value="1",
        )

        if self.config.generator.name == "openai":
            api.add_space_secret(
                repo_id=self.config.hub.repo_id,
                key="OPENAI_API_KEY",
                value=os.environ[self.config.generator.api_key_variable_name],
            )

        folders_to_upload: list[Path] = [
            Path("src"),
            Path("config"),
            Path(self.config.dirs.data) / self.config.dirs.processed,
            Path(self.config.dirs.data) / self.config.dirs.final,
        ]
        files_to_upload: list[Path] = [Path("Dockerfile"), Path("pyproject.toml")]
        for path in folders_to_upload + files_to_upload:
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist. Please create it.")

        for folder in folders_to_upload:
            api.upload_folder(
                repo_id=self.config.hub.repo_id,
                repo_type="space",
                folder_path=str(folder),
                path_in_repo=str(folder),
                commit_message=f"Upload {folder!r} folder to the hub.",
            )

        for file in files_to_upload:
            api.upload_file(
                repo_id=self.config.hub.repo_id,
                repo_type="space",
                path_or_fileobj=str(file),
                path_in_repo=str(file),
                commit_message=f"Upload {file!r} script to the hub.",
            )

        logger.info(
            f"Pushed the demo to the hub! You can access it at "
            f"https://hf.co/spaces/{self.config.demo.persistent_sharing.repo_id}."
        )

    def close(self) -> None:
        """Close the demo."""
        if self.demo:
            self.demo.close()
            logger.info("Closed the demo.")

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
            data:
                The like data.
            history:
                The chat history.
        """
        if data.liked:
            logger.info("User liked the response.")
        else:
            logger.info("User disliked the response.")

        retrieved_document_data = dict(
            id=json.dumps(
                [getattr(document, "id") for document in self.retrieved_documents]
            )
        )
        record = {
            "query": history[-2][0],
            "response": history[-1][1],
            "liked": int(data.liked),
        } | retrieved_document_data

        # Add the record to the table "feedback" in the database.
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute(
            ("INSERT INTO feedback VALUES (:query, :response, :liked, :id)"), record
        )
        self.connection.commit()
        self.connection.close()

        logger.info("Recorded the vote in the database.")
