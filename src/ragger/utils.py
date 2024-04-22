"""Utility functions for the project."""

import torch


def get_device() -> str:
    """This returns the device to use for the project.

    Returns:
        The device to use for the project.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
