"""Exceptions used by astartes"""


class NotInitializedError(AttributeError):
    """Used when a class is called without proper initialization."""

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class DatasetError(ValueError):
    """Used when a sampler runs out of data."""

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class InvalidAstartesConfigurationError(ValueError):
    """Used when astartes train_test_split if called in an invalid manner."""

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class MoleculesNotInstalledError(RuntimeError):
    """Used when attempting to featurize molecules without install."""

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)
