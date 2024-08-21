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
            f"Missing package(s): {', '.join(package_names)}. Please install them using "
            "e.g., `pip install <package_name>`."
        )


class MissingExtra(Exception):
    """Exception raised when an extra is missing."""

    def __init__(self, extras: list[str]) -> None:
        """Initialise the exception.

        Args:
            extras:
                The names of the missing extras.
        """
        self.extras = extras
        super().__init__(
            f"Missing extra(s): {', '.join(extras)}. Please install them using "
            "e.g., `pip install ragger[<extra_name>]`."
        )
