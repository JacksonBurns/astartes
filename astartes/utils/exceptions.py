"""Exceptions used by astartes"""


class MoleculesNotInstalledError(RuntimeError):  # pragma: no cover
    """Used when attempting to featurize molecules without install."""

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class InvalidConfigurationError(RuntimeError):
    """Used when user-requested split/data would not work."""

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class SamplerNotImplementedError(RuntimeError):
    """Used when attempting to call a non-existent sampler."""

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)
