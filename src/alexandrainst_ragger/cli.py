"""Command-line interface for the `alexandrainst_ragger` package."""

import logging
from pathlib import Path

import click

from .demo import Demo
from .rag_system import RagSystem

logger = logging.getLogger(__package__)


@click.command()
@click.option(
    "--config-file",
    "-c",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the configuration file, which should be a JSON or YAML file.",
)
def run_demo(config_file: Path | None) -> None:
    """Run a RAG demo.

    Args:
        config_file:
            Path to the configuration file.
    """
    logger.info(f"Running the RAG demo with config file '{config_file}'.")
    rag_system = RagSystem.from_config(config_file=config_file)
    demo = Demo.from_config(rag_system=rag_system, config_file=config_file)
    demo.launch()


@click.command()
@click.option(
    "--config-file",
    "-c",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the configuration file, which should be a JSON or YAML file.",
)
def compile(config_file: Path) -> None:
    """Compile a RAG system.

    Args:
        config_file:
            Path to the configuration file.
    """
    logger.info(f"Compiling the RAG system with config file '{config_file}'.")
    RagSystem.from_config(config_file=config_file)
    logger.info("RAG system compiled successfully.")
