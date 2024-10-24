"""Unit tests for the `data_models` module."""

from abc import ABC
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

import ragger.data_models


def test_all_classes_are_either_pydantic_models_or_abcs():
    """Test that all classes in `data_models` are either Pydantic models or ABCs."""
    accepted_exceptions = [Path, ConfigDict, Field, str]
    classes = [
        cls
        for cls in vars(ragger.data_models).values()
        if isinstance(cls, type) and cls not in accepted_exceptions
    ]
    for cls in classes:
        assert issubclass(cls, BaseModel) or issubclass(cls, ABC)
