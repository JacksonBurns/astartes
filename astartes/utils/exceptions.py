"""Exceptions used by astartes"""


class MoleculesNotInstalledError(RuntimeError):
    """Used when attempting to featurize molecules without install."""

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)
