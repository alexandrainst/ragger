"""A Gradio demo of the RAG system."""

import json
import logging
import os
import sqlite3
import typing
import warnings
from pathlib import Path

from .constants import (
    DANISH_DEMO_TITLE,
    DANISH_DESCRIPTION,
    DANISH_FEEDBACK_INSTRUCTION,
    DANISH_INPUT_BOX_PLACEHOLDER,
    DANISH_NO_DOCUMENTS_REPLY,
    DANISH_SUBMIT_BUTTON_VALUE,
    DANISH_THANK_YOU_FEEDBACK,
    ENGLISH_DEMO_TITLE,
    ENGLISH_DESCRIPTION,
    ENGLISH_FEEDBACK_INSTRUCTION,
    ENGLISH_INPUT_BOX_PLACEHOLDER,
    ENGLISH_NO_DOCUMENTS_REPLY,
    ENGLISH_SUBMIT_BUTTON_VALUE,
    ENGLISH_THANK_YOU_FEEDBACK,
)
from .data_models import Document, EmbeddingStore, PersistentSharingConfig
from .generator import OpenAIGenerator
from .rag_system import RagSystem
from .utils import format_answer, is_installed, load_config, raise_if_not_installed

if is_installed(package_name="gradio"):
    import gradio as gr

if is_installed(package_name="huggingface_hub"):
    from huggingface_hub import CommitScheduler, HfApi
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

if typing.TYPE_CHECKING:
    import gradio as gr
    from huggingface_hub import CommitScheduler, HfApi
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

Message = str | None
Exchange = tuple[Message, Message]
History = list[Exchange]


logger = logging.getLogger(__package__)


class Demo:
    """A Gradio demo of the RAG system."""

    def __init__(
        self,
        rag_system: RagSystem,
        feedback_db_path: Path = Path("feedback.db"),
        feedback_mode: typing.Literal[
            "strict-feedback", "feedback", "no-feedback"
        ] = "feedback",
        gradio_theme: str = "monochrome",
        title: str | None = None,
        description: str | None = None,
        feedback_instruction: str | None = None,
        thank_you_feedback: str | None = None,
        input_box_placeholder: str | None = None,
        submit_button_value: str | None = None,
        no_documents_reply: str | None = None,
        persistent_sharing_config: PersistentSharingConfig | None = None,
        host: str | None = None,
        port: int = 7860,
    ) -> None:
        """Initialise the demo.

        Args:
            rag_system:
                The RAG system to use.
            feedback_db_path (optional):
                The path to the feedback database. Defaults to "feedback.db".
            feedback_mode (optional):
                The feedback mode to use. Can be "strict-feedback", "feedback", or
                "no-feedback". Defaults to "strict-feedback".
            gradio_theme (optional):
                The Gradio theme to use. Defaults to "monochrome".
            title (optional):
                The title of the demo. If None then a default title is used, based on
                the language of the RAG system. Defaults to None.
            description (optional):
                The description of the demo. If None then a default description is used,
                based on the language of the RAG system. Defaults to None.
            feedback_instruction (optional):
                The instruction to display to the user after they have submitted
                feedback. If None then a default instruction is used, based on the
                language of the RAG system. Defaults to None.
            thank_you_feedback (optional):
                The message to display to the user after they have submitted feedback.
                If None then a default message is used, based on the language of the RAG
                system. Defaults to None.
            input_box_placeholder (optional):
                The placeholder text to display in the input box. If None then a default
                placeholder is used, based on the language of the RAG system. Defaults
                to None.
            submit_button_value (optional):
                The value to display on the submit button. If None then a default value
                is used, based on the language of the RAG system. Defaults to None.
            no_documents_reply (optional):
                The reply to use when no documents are found. If None then a default
                reply is used, based on the language of the RAG system. Defaults to None.
            persistent_sharing_config (optional):
                The configuration for persistent sharing of the demo. If None then no
                persistent sharing is used. Defaults to None.
            host (optional):
                The host to use. Defaults to None, which uses the value of the
                GRADIO_SERVER_NAME environment variable, or "localhost" if that is not
                set.
            port (optional):
                The port to use. Defaults to 7860.
        """
        raise_if_not_installed(
            package_names=["gradio", "huggingface_hub"], extra="demo"
        )

        title_mapping = dict(da=DANISH_DEMO_TITLE, en=ENGLISH_DEMO_TITLE)
        description_mapping = dict(da=DANISH_DESCRIPTION, en=ENGLISH_DESCRIPTION)
        feedback_instruction_mapping = dict(
            da=DANISH_FEEDBACK_INSTRUCTION, en=ENGLISH_FEEDBACK_INSTRUCTION
        )
        thank_you_feedback_mapping = dict(
            da=DANISH_THANK_YOU_FEEDBACK, en=ENGLISH_THANK_YOU_FEEDBACK
        )
        input_box_placeholder_mapping = dict(
            da=DANISH_INPUT_BOX_PLACEHOLDER, en=ENGLISH_INPUT_BOX_PLACEHOLDER
        )
        submit_button_value_mapping = dict(
            da=DANISH_SUBMIT_BUTTON_VALUE, en=ENGLISH_SUBMIT_BUTTON_VALUE
        )
        no_documents_reply_mapping = dict(
            da=DANISH_NO_DOCUMENTS_REPLY, en=ENGLISH_NO_DOCUMENTS_REPLY
        )

        self.rag_system = rag_system
        self.feedback_db_path = feedback_db_path
        self.feedback_mode = feedback_mode
        self.gradio_theme = gradio_theme
        self.title = title or title_mapping[rag_system.language]
        self.description = description or description_mapping[rag_system.language]
        self.feedback_instruction = (
            feedback_instruction or feedback_instruction_mapping[rag_system.language]
        )
        self.thank_you_feedback = (
            thank_you_feedback or thank_you_feedback_mapping[rag_system.language]
        )
        self.input_box_placeholder = (
            input_box_placeholder or input_box_placeholder_mapping[rag_system.language]
        )
        self.submit_button_value = (
            submit_button_value or submit_button_value_mapping[rag_system.language]
        )
        self.no_documents_reply = (
            no_documents_reply or no_documents_reply_mapping[rag_system.language]
        )
        self.persistent_sharing_config = persistent_sharing_config
        self.host = host or os.getenv("GRADIO_SERVER_NAME", "localhost")
        self.port = port

        # Ensure the database file exists
        self.feedback_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_db_path.touch(exist_ok=True)

        match self.feedback_mode:
            case "strict-feedback" | "feedback":
                logger.info(f"Using the {self.feedback_mode!r} feedback mode.")
                with sqlite3.connect(self.feedback_db_path) as connection:
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
            case "no-feedback":
                logger.info("No feedback will be collected.")
            case _:
                raise ValueError(
                    "The feedback mode must be one of 'strict-feedback', 'feedback', "
                    "or 'no-feedback'."
                )

        # This will only run when the demo is running in a Hugging Face Space
        if os.getenv("RUNNING_IN_SPACE") == "1" and self.persistent_sharing_config:
            logger.info("Running in a Hugging Face space.")

            # Suppress warnings when running in a Hugging Face space, as this causes
            # the space to crash
            warnings.filterwarnings(action="ignore")

            # Initialise commit scheduler, which will commit files to the Hub at
            # regular intervals
            if self.feedback_mode in {"strict-feedback", "feedback"}:
                backup_dir = self.feedback_db_path.parent
                db_repo_id = self.persistent_sharing_config.database_repo_id
                every = self.persistent_sharing_config.database_update_frequency
                assert backup_dir.exists(), f"{backup_dir!r} does not exist!"
                self.scheduler = CommitScheduler(
                    repo_id=db_repo_id,
                    repo_type="dataset",
                    folder_path=backup_dir,
                    path_in_repo=str(backup_dir),
                    squash_history=True,
                    every=every,
                    token=os.getenv(
                        self.persistent_sharing_config.hf_token_variable_name
                    ),
                    private=True,
                )
                logger.info(
                    "Initialised the commit scheduler, which will backup the feedback "
                    f"database to {db_repo_id} every {every:,} minutes."
                )

        self.retrieved_documents: list[Document] = list()
        self.blocks: gr.Blocks | None = None

    @classmethod
    def from_config(
        cls, rag_system: RagSystem, config_file: str | Path | None
    ) -> "Demo":
        """Create a demo from a configuration.

        Args:
            rag_system:
                The RAG system.
            config_file:
                Path to the configuration file.

        Returns:
            The demo.
        """
        config = load_config(config_file=config_file)

        kwargs: dict[str, typing.Any] = dict(rag_system=rag_system)
        if "feedback_db_path" in config:
            kwargs["feedback_db_path"] = Path(config["feedback_db_path"])
        if "feedback_mode" in config:
            kwargs["feedback_mode"] = config["feedback_mode"]
        if "gradio_theme" in config:
            kwargs["gradio_theme"] = config["gradio_theme"]
        if "title" in config:
            kwargs["title"] = config["title"]
        if "description" in config:
            kwargs["description"] = config["description"]
        if "feedback_instruction" in config:
            kwargs["feedback_instruction"] = config["feedback_instruction"]
        if "thank_you_feedback" in config:
            kwargs["thank_you_feedback"] = config["thank_you_feedback"]
        if "input_box_placeholder" in config:
            kwargs["input_box_placeholder"] = config["input_box_placeholder"]
        if "submit_button_value" in config:
            kwargs["submit_button_value"] = config["submit_button_value"]
        if "no_documents_reply" in config:
            kwargs["no_documents_reply"] = config["no_documents_reply"]
        if "persistent_sharing_config" in config:
            kwargs["persistent_sharing_config"] = PersistentSharingConfig(
                **config["persistent_sharing_config"]
            )
        if "host" in config:
            kwargs["host"] = config["host"]
        if "port" in config:
            kwargs["port"] = config["port"]
        return cls(**kwargs)

    def build_demo(self) -> "gr.Blocks":
        """Build the demo.

        Returns:
            The demo.
        """
        logger.info("Building the demo...")
        with gr.Blocks(
            theme=self.gradio_theme, title=self.title, fill_height=True
        ) as demo:
            gr.components.HTML(f"<center><h1>{self.title}</h1></center>")
            directions = gr.components.HTML(
                f"<b><center>{self.description}</b></center>", label="p"
            )
            chatbot = gr.Chatbot(
                value=[],
                elem_id="chatbot",
                bubble_full_width=False,
                scale=1,
                type="tuples",
            )
            with gr.Row():
                input_box = gr.Textbox(
                    scale=4,
                    show_label=False,
                    placeholder=self.input_box_placeholder,
                    container=False,
                )
            submit_button = gr.Button(value=self.submit_button_value, variant="primary")
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

            if self.feedback_mode in ["strict-feedback", "feedback"]:
                submit_button_has_added_text_and_asked.then(
                    fn=lambda: gr.update(
                        value=f"""
                            <b><center>
                            {self.feedback_instruction}
                            </center></b>
                        """
                    ),
                    outputs=[directions],
                    queue=False,
                )

                input_box_has_added_text_and_asked.then(
                    fn=lambda: gr.update(
                        value=f"""
                            <b><center>
                            {self.feedback_instruction}
                            </center></b>
                        """
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
                            "<b><center>" f"{self.thank_you_feedback}" "</center></b>"
                        )
                    ),
                    outputs=[directions],
                    queue=False,
                )
        logger.info("Built the demo.")
        return demo

    def launch(self) -> None:
        """Launch the demo."""
        self.blocks = self.build_demo()
        assert self.blocks is not None

        # If we are storing the demo persistently we push it to the Hugging Face Hub,
        # unless we are already running this from the Hub
        if (
            self.persistent_sharing_config is not None
            and os.getenv("RUNNING_IN_SPACE") != "1"
        ):
            self.push_to_hub()
            return

        logger.info(f"Launching the demo at {self.host}:{self.port}...")
        self.blocks.queue().launch(server_name=self.host, server_port=self.port)

    def push_to_hub(self) -> None:
        """Pushes the demo to a Hugging Face Space on the Hugging Face Hub."""
        if self.persistent_sharing_config is None:
            raise ValueError(
                "The demo must be shared persistently to push it to the hub. Please "
                "set the `persistent_sharing_config` field and try again."
            )

        logger.info("Pushing the demo to the hub...")

        api = HfApi(
            token=os.getenv(self.persistent_sharing_config.hf_token_variable_name, True)
        )

        if not api.repo_exists(repo_id=self.persistent_sharing_config.space_repo_id):
            api.create_repo(
                repo_id=self.persistent_sharing_config.space_repo_id,
                repo_type="space",
                space_sdk="docker",
                exist_ok=True,
                private=True,
            )
            logger.info("Created the space on the hub.")

        # This environment variable is used to trigger the creation of a commit
        # scheduler when the demo is initialised, which will commit the final data
        # directory to the Hub at regular intervals.
        api.add_space_variable(
            repo_id=self.persistent_sharing_config.space_repo_id,
            key="RUNNING_IN_SPACE",
            value="1",
        )

        api.add_space_secret(
            repo_id=self.persistent_sharing_config.space_repo_id,
            key=self.persistent_sharing_config.hf_token_variable_name,
            value=os.environ[self.persistent_sharing_config.hf_token_variable_name],
        )
        if isinstance(self.rag_system.generator, OpenAIGenerator):
            api.add_space_secret(
                repo_id=self.persistent_sharing_config.space_repo_id,
                key="OPENAI_API_KEY",
                value=self.rag_system.generator.api_key or os.environ["OPENAI_API_KEY"],
            )

        logger.info("Added environment variables and secrets to the space.")

        # The feedback database is stored in a separate repo, so we need to pull the
        # newest version of the database before pushing the demo to the hub
        if self.feedback_mode in {"strict-feedback", "feedback"}:
            try:
                api.hf_hub_download(
                    repo_id=self.persistent_sharing_config.database_repo_id,
                    repo_type="dataset",
                    filename=str(self.feedback_db_path),
                    force_download=True,
                    local_dir=".",
                )
                logger.info(
                    "Downloaded the feedback database from the Hugging Face Hub."
                )
            # If the database or database repo does not exist, we skip this step
            except (ValueError, EntryNotFoundError, RepositoryNotFoundError):
                logger.info(
                    "The feedback database or database repo does not exist. Skipping."
                )

        folders_to_upload: list[Path] = [Path("src")]
        files_to_upload: list[Path] = [
            Path("Dockerfile"),
            Path("pyproject.toml"),
            Path("uv.lock"),
        ]

        if (document_store_path := self.rag_system.document_store.path) is not None:
            if document_store_path.is_dir():
                folders_to_upload.append(document_store_path)
            else:
                files_to_upload.append(document_store_path)

        if hasattr(self.rag_system.retriever, "embedding_store"):
            embedding_store: EmbeddingStore = self.rag_system.retriever.embedding_store
            if (embedding_store_path := embedding_store.path) is not None:
                if embedding_store_path.is_dir():
                    folders_to_upload.append(embedding_store_path)
                else:
                    files_to_upload.append(embedding_store_path)

        for path in folders_to_upload + files_to_upload:
            if not path.exists():
                raise FileNotFoundError(
                    f"{str(path)!r} does not exist. Please create it."
                )

        for folder in folders_to_upload:
            logger.info(f"Uploading {str(folder)!r} folder to the hub...")
            api.upload_folder(
                repo_id=self.persistent_sharing_config.space_repo_id,
                repo_type="space",
                folder_path=str(folder),
                path_in_repo=str(folder),
                commit_message=f"Upload {str(folder)!r} folder to the hub.",
            )

        for path in files_to_upload:
            logger.info(f"Uploading {str(path)!r} to the hub...")
            api.upload_file(
                repo_id=self.persistent_sharing_config.space_repo_id,
                repo_type="space",
                path_or_fileobj=str(path),
                path_in_repo=str(path),
                commit_message=f"Upload {str(path)!r} script to the hub.",
            )

        logger.info(
            f"Pushed the demo to the hub! You can access it at "
            f"https://hf.co/spaces/{self.persistent_sharing_config.space_repo_id}."
        )

    def close(self) -> None:
        """Close the demo."""
        if self.blocks:
            self.blocks.close()
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
        if self.feedback_mode == "strict-feedback":
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
                formatted_answer = format_answer(
                    answer=generated_answer,
                    documents=documents,
                    no_documents_reply=self.no_documents_reply,
                )
                history[-1] = (None, formatted_answer)
                yield history
        else:
            generated_answer, documents = answer_or_stream
        generated_answer = format_answer(
            answer=generated_answer,
            documents=documents,
            no_documents_reply=self.no_documents_reply,
        )
        self.retrieved_documents = documents
        history[-1] = (None, generated_answer)
        yield history

    def vote(self, data: "gr.LikeData", history: History):
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
        with sqlite3.connect(self.feedback_db_path) as connection:
            connection.execute(
                "INSERT INTO feedback VALUES (:query, :response, :liked, :id)", record
            )
            connection.commit()

        logger.info("Recorded the vote in the database.")
