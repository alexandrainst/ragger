"""Command-line interface for the `ragger` package."""

from pathlib import Path

import click
import yaml

from .demo import Demo
from .rag_system import RagSystem


@click.command()
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
def run_demo(config_file: str) -> None:
    """Run a RAG demo.

    Args:
        config_file:
            Path to the configuration file.
    """
    with Path(config_file).open("r") as f:
        config = yaml.safe_load(f)

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
    with Path(config_file).open("r") as f:
        config = yaml.safe_load(f)
    RagSystem.from_config(config=config)
