"""Utility constants and functions used in the project."""

import importlib.util
import json
import logging
import re
from pathlib import Path

import yaml

from .data_models import Document
from .exceptions import MissingExtra, MissingPackage

logger = logging.getLogger(__package__)


def format_answer(
    answer: str, documents: list[Document], no_documents_reply: str
) -> str:
    """Format the answer as HTML with the relevant documents.

    Args:
        answer:
            The generated answer.
        documents:
            The relevant documents.
        no_documents_reply:
            The reply to use when no documents are found.

    Returns:
        The formatted answer.
    """
    match len(documents):
        case 0:
            answer = no_documents_reply
        case 1:
            answer += "<br><br>Kilde:<br>"
        case _:
            answer += "<br><br>Kilder:<br>"

    formatted_ids = [
        f"<a href='{document.id}'>{document.id}</a>"
        if is_link(text=document.id)
        else document.id
        for document in documents
    ]

    answer += "<br>".join(
        f"<details><summary>{formatted_id}</summary>{document.text}</details>"
        for formatted_id, document in zip(formatted_ids, documents)
    )
    return answer


def is_link(text: str) -> bool:
    """Check if the text is a link.

    Args:
        text:
            The text to check.

    Returns:
        Whether the text is a link.
    """
    url_regex = (
        r"^(https?:\/\/)?"  # Begins with http:// or https://, or neither
        r"(\w+\.)+"  # Then one or more blocks of lower-case letters and a dot
        r"\w{2,4}"  # Then two to four lower-case letters (e.g., .com, .dk, .org)
        r"(\/#?\w+)*?"  # Optionally followed by subdirectories or anchors
        r"(\/\w+\.\w{1,4})?"  # Optionally followed by a file suffix (e.g., .html)
    )
    return re.match(pattern=url_regex, string=text) is not None


def get_device() -> str:
    """Get the device to use for computation."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def is_installed(package_name: str) -> bool:
    """Check if a package is installed.

    Args:
        package_name:
            The name of the package to check for.

    Returns:
        Whether the package is installed.
    """
    return importlib.util.find_spec(name=package_name) is not None


def raise_if_not_installed(
    package_names: list[str],
    extra: str | list[str] | None = None,
    installation_alias_mapping: dict[str, str] | None = None,
) -> None:
    """Raise an exception if any of the packages are not installed.

    Args:
        package_names:
            The names of the packages to check for.
        extra:
            The name of the extra corresponding to the packages. Can also be a list if
            the package belongs to either of multiple extras. Can be None if no extra
            is needed.
        installation_alias_mapping:
            A mapping from package names to installation aliases, if a package is not
            installed with `pip install PACKAGE_NAME`. Can be None if no aliases are
            needed.

    Raises:
        MissingExtra:
            If any of the extras are not installed.
        MissingPackage:
            If any of the packages are not installed.
    """
    if installation_alias_mapping is None:
        installation_alias_mapping = dict()

    # Check that all the packages in the mapping are present in the package names
    assert all(package in package_names for package in installation_alias_mapping)

    missing_packages = [
        package for package in package_names if not is_installed(package_name=package)
    ]
    if missing_packages:
        if extra is not None:
            raise MissingExtra(extra=extra)
        missing_packages = [
            installation_alias_mapping.get(package, package)
            for package in missing_packages
        ]
        raise MissingPackage(package_names=missing_packages)


def load_config(config_file: str | Path | None) -> dict:
    """Load a configuration file.

    Args:
        config_file:
            Path to the configuration file.

    Returns:
        The configuration.
    """
    config_file = Path(config_file) if config_file is not None else None

    if config_file is None:
        logger.warning("No configuration file provided. Using default configuration.")
        return dict()

    if config_file.read_text().strip() == "":
        logger.warning(
            "Empty configuration file provided. Using default configuration."
        )
        return dict()

    with config_file.open("r") as f:
        if config_file.suffix == ".json":
            return json.load(f)
        elif config_file.suffix == ".yaml":
            return yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported file extension: {config_file.suffix}. Please provide "
                "a JSON or YAML file."
            )
