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
