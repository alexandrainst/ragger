"""Test fixtures used throughout the test suite."""

import typing
from tempfile import NamedTemporaryFile

import pytest
from omegaconf import DictConfig
from ragger.data_models import Document


@pytest.fixture(scope="session")
def system_prompt() -> typing.Generator[str, None, None]:
    """A system prompt for the generator."""
    yield """
    Du er en seniorkonsulent, som har til opgave at finde svar på spørgsmål ud fra en
    række dokumenter.

    Du vil altid referere til ID'erne på de dokumenter, som indeholder svaret, og *kun*
    disse dokumenter. Du svarer altid på dansk.

    Dit svar er i JSON-format, med keys "answer" og "sources" i din JSON dictionary.
    Her er "answer" dit svar, og "sources" er en liste af ID'er på de dokumenter, som
    du baserer dit svar på.
    """


@pytest.fixture(scope="session")
def prompt() -> typing.Generator[str, None, None]:
    """A prompt for the generator."""
    yield """
    Her er en række dokumenter, som du skal basere din besvarelse på.

    <documents>
    {documents}
    </documents>

    Ud fra disse dokumenter, hvad er svaret på følgende spørgsmål?

    <question>
    {query}
    </question>

    Husk at du altid skal svare på dansk.

    <answer>
    """


@pytest.fixture(scope="session")
def dirs_params() -> typing.Generator[dict, None, None]:
    """Parameters for the directories."""
    yield dict(
        data="data", raw="raw", processed="processed", final="final", models="models"
    )


@pytest.fixture(scope="session")
def documents() -> typing.Generator[list[Document], None, None]:
    """Some documents for testing."""
    yield [
        Document(id="1", text="Den hvide kat hedder Sjusk."),
        Document(id="2", text="Den sorte kat hedder Sutsko."),
        Document(id="3", text="Den røde kat hedder Pjuskebusk."),
        Document(id="4", text="Den grønne kat hedder Sjask."),
        Document(id="5", text="Den blå kat hedder Sky."),
    ]


@pytest.fixture(scope="session")
def document_store_params(documents) -> typing.Generator[dict, None, None]:
    """Parameters for the document store."""
    with NamedTemporaryFile(mode="w", suffix=".jsonl") as file:
        data_str = "\n".join(document.model_dump_json() for document in documents)
        file.write(data_str)
        file.flush()
        yield dict(name="jsonl", filename=file.name)


@pytest.fixture(scope="session")
def embedder_params() -> typing.Generator[dict, None, None]:
    """Parameters for the embedder."""
    yield dict(
        name="e5", model_id="intfloat/multilingual-e5-small", document_text_field="text"
    )


@pytest.fixture(scope="session")
def embedding_store_params() -> typing.Generator[dict, None, None]:
    """Parameters for the embedding store."""
    yield dict(name="numpy", num_documents_to_retrieve=2, filename=None)


@pytest.fixture(scope="session")
def generator_params(system_prompt, prompt) -> typing.Generator[dict, None, None]:
    """Parameters for the generator."""
    yield dict(
        name="openai",
        api_key_variable_name="OPENAI_API_KEY",
        model="gpt-3.5-turbo-0125",
        temperature=0.0,
        stream=False,
        timeout=60,
        max_tokens=128,
        system_prompt=system_prompt,
        prompt=prompt,
    )


@pytest.fixture(scope="session")
def demo_params() -> typing.Generator[dict, None, None]:
    """Parameters for the demo."""
    with NamedTemporaryFile(mode="w", suffix=".db") as file:
        yield dict(
            name="danish",
            host="localhost",
            port=7860,
            share=False,
            password_protected=False,
            feedback="no-feedback",
            db_path=file.name,
            title="En titel",
            feedback_instructions="Skriv din feedback her:",
            thank_you_feedback="Tak for din feedback!",
            input_box_placeholder="Skriv din tekst her...",
            submit_button_value="Send",
            no_documents_reply="Jeg kunne desværre ikke finde noget svar.",
            description="En beskrivelse",
        )


@pytest.fixture(scope="session")
def full_config(
    dirs_params,
    document_store_params,
    embedder_params,
    embedding_store_params,
    generator_params,
    demo_params,
) -> typing.Generator[DictConfig, None, None]:
    """A valid Hydra configuration."""
    yield DictConfig(
        dict(
            dirs=dirs_params,
            document_store=document_store_params,
            embedder=embedder_params,
            embedding_store=embedding_store_params,
            generator=generator_params,
            demo=demo_params,
            verbose=False,
        )
    )
