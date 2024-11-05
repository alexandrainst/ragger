"""Exceptions in the project."""


class MissingPackage(Exception):
    """Exception raised when a package is missing."""

    def __init__(self, package_names: list[str]) -> None:
        """Initialise the exception.

        Args:
            package_names:
                The names of the missing packages.
        """
        self.package_names = package_names
        super().__init__(
            f"Missing package(s): {', '.join(package_names)}. Please install them "
            f"using , e.g., `pip install {' '.join(package_names)}`."
        )


class MissingExtra(Exception):
    """Exception raised when an extra is missing."""

    def __init__(self, extra: str | list[str]) -> None:
        """Initialise the exception.

        Args:
            extra:
                The names of the missing extra, or multiple extras if one of them is
                needed.
        """
        extra = extra[0] if isinstance(extra, list) and len(extra) == 1 else extra
        if isinstance(extra, str):
            super().__init__(
                f"Missing extra: {extra}. Please install it using , e.g., "
                f"`pip install alexandrainst_ragger[{extra}]`."
            )
        else:
            super().__init__(
                f"Missing one of the following extras: {', '.join(extra)}. Please "
                "choose one and install it using , e.g., `pip install "
                "alexandrainst_ragger[EXTRA]`."
            )
