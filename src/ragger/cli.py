"""Command-line interface for the `ragger` package."""

import logging
from pathlib import Path

import click

from ragger.utils import load_config

from .demo import Demo
from .rag_system import RagSystem

logger = logging.getLogger(__package__)


@click.command()
@click.option(
    "--config_file",
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
    config = load_config(config_file=config_file)
    rag_system = RagSystem.from_config(config=config)
    demo = Demo.from_config(rag_system=rag_system, config=config)
    demo.launch()


@click.command()
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
def compile(config_file: str) -> None:
    """Compile a RAG system.

    Args:
        config_file:
            Path to the configuration file.
    """
    config = load_config(config_file=Path(config_file))
    RagSystem.from_config(config=config)
