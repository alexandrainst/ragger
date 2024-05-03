"""A Gradio demo of the RAG system."""

import json
import logging
import os
import sqlite3
import typing
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import gradio as gr
import huggingface_hub
from huggingface_hub import CommitScheduler, HfApi
from huggingface_hub.utils import LocalTokenNotFoundError
from omegaconf import DictConfig, OmegaConf

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

        # This will only run when the demo is running in a Hugging Face Space
        if os.getenv("RUNNING_IN_SPACE") == "1":
            logger.info("Running in a Hugging Face space.")

            # Suppress warnings when running in a Hugging Face space, as this causes
            # the space to crash
            warnings.filterwarnings(action="ignore")

            # Initialise commit scheduler, which will commit files to the Hub at
            # regular intervals
            final_data_path = Path(self.config.dirs.data) / self.config.dirs.final
            assert final_data_path.exists(), f"{final_data_path!r} does not exist!"
            self.scheduler = CommitScheduler(
                repo_id=self.config.demo.persistent_sharing.database_repo_id,
                repo_type="dataset",
                folder_path=final_data_path,
                path_in_repo=str(final_data_path),
                squash_history=True,
                every=5,
                token=os.getenv(
                    self.config.demo.persistent_sharing.token_variable_name
                ),
                private=True,
            )

        self.retrieved_documents: list[Document] = []
        self.rag_system = RagSystem(config=config)

        self.db_path = Path(config.dirs.data) / config.demo.db_path
        match self.config.demo.mode:
            case "strict_feedback" | "feedback":
                logger.info(f"Using the {self.config.demo.mode!r} feedback mode.")
                with sqlite3.connect(self.db_path) as connection:
                    table_empty = not connection.execute("""
                        SELECT name FROM sqlite_master
                        WHERE type='table' AND name='feedback'
                    """).fetchone()
                    if table_empty:
                        connection.execute("""
                            CREATE TABLE feedback (query text, response text,
                            liked integer, document_ids text)
                        """)
                        connection.commit()
            case "no_feedback":
                logger.info("No feedback will be collected.")
            case _:
                raise ValueError(
                    "The feedback mode must be one of 'strict_feedback', 'feedback', "
                    "or 'no_feedback'."
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
        if self.config.demo.share not in {"no-share", "temporary", "persistent"}:
            raise ValueError(
                "The `demo.share` field in the config must be one of 'temporary', "
                "'persistent', or 'no-share'. It is currently set to "
                f"{self.config.demo.share!r}. Please change it and try again."
            )

        self.demo = self.build_demo()

        # If we are storing the demo persistently we push it to the Hugging Face Hub,
        # unless we are already running this from the Hub
        if (
            self.config.demo.share == "persistent"
            and os.getenv("RUNNING_IN_SPACE") != "1"
        ):
            self.push_to_hub()
            return

        logger.info("Launching the demo...")

        launch_kwargs = dict(
            server_name=self.config.demo.host, server_port=self.config.demo.port
        )

        # Add password protection to the demo, if required
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
        launch_kwargs["auth"] = auth

        if self.config.demo.share == "temporary":
            launch_kwargs["share"] = True

        self.demo.queue().launch(**launch_kwargs)

    def push_to_hub(self) -> None:
        """Pushes the demo to a Hugging Face Space on the Hugging Face Hub."""
        if self.config.demo.share != "persistent":
            raise ValueError(
                "The demo must be shared persistently to push it to the hub. Please "
                "set the `demo.share` field to 'persistent' in the config and try "
                "again."
            )

        space_repo_id = self.config.demo.persistent_sharing.space_repo_id
        if space_repo_id is None:
            raise ValueError(
                "The `demo.persistent_sharing.space_repo_id` field must be set in the "
                "config to push the demo to the hub. Please set it and try again."
            )

        database_repo_id = self.config.demo.persistent_sharing.database_repo_id
        if database_repo_id is None:
            raise ValueError(
                "The `demo.persistent_sharing.database_repo_id` field must be set in "
                "the config to push the demo to the hub. Please set it and try again."
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
            token=os.getenv(
                self.config.demo.persistent_sharing.token_variable_name, True
            )
        )

        if not api.repo_exists(repo_id=space_repo_id):
            api.create_repo(
                repo_id=space_repo_id,
                repo_type="space",
                space_sdk="docker",
                exist_ok=True,
                private=True,
            )

        # This environment variable is used to trigger the creation of a commit
        # scheduler when the demo is initialised, which will commit the final data
        # directory to the Hub at regular intervals.
        api.add_space_variable(repo_id=space_repo_id, key="RUNNING_IN_SPACE", value="1")

        api.add_space_secret(
            repo_id=space_repo_id,
            key=self.config.demo.persistent_sharing.token_variable_name,
            value=os.environ[self.config.demo.persistent_sharing.token_variable_name],
        )
        if self.config.generator.name == "openai":
            api.add_space_secret(
                repo_id=space_repo_id,
                key=self.config.generator.api_key_variable_name,
                value=os.environ[self.config.generator.api_key_variable_name],
            )

        # Upload config separately, as the user might have created overrides when
        # running this current session
        with TemporaryDirectory() as temp_dir:
            api.upload_folder(
                repo_id=space_repo_id,
                repo_type="space",
                folder_path=temp_dir,
                path_in_repo="config",
                commit_message="Create config folder.",
            )
        with NamedTemporaryFile(mode="w", suffix=".yaml") as file:
            config_yaml = OmegaConf.to_yaml(cfg=self.config)
            file.write(config_yaml)
            file.flush()
            api.upload_file(
                repo_id=space_repo_id,
                repo_type="space",
                path_or_fileobj=file.name,
                path_in_repo="config/ragger_config.yaml",
                commit_message="Upload config to the hub.",
            )

        folders_to_upload: list[Path] = [
            Path("src"),
            Path(self.config.dirs.data) / self.config.dirs.processed,
            Path(self.config.dirs.data) / self.config.dirs.final,
        ]
        files_to_upload: list[Path] = [
            Path("Dockerfile"),
            Path("pyproject.toml"),
            Path("poetry.lock"),
        ]
        for path in folders_to_upload + files_to_upload:
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist. Please create it.")

        for folder in folders_to_upload:
            api.upload_folder(
                repo_id=space_repo_id,
                repo_type="space",
                folder_path=str(folder),
                path_in_repo=str(folder),
                commit_message=f"Upload {folder!r} folder to the hub.",
            )

        for path in files_to_upload:
            api.upload_file(
                repo_id=space_repo_id,
                repo_type="space",
                path_or_fileobj=str(path),
                path_in_repo=str(path),
                commit_message=f"Upload {path!r} script to the hub.",
            )

        logger.info(
            f"Pushed the demo to the hub! You can access it at "
            f"https://hf.co/spaces/{space_repo_id}."
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
            logger.info(f"User liked the response {data.value!r}.")
        else:
            logger.info(f"User disliked the response {data.value!r}.")

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
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO feedback VALUES (:query, :response, :liked, :id)", record
            )
            connection.commit()

        logger.info("Recorded the vote in the database.")
