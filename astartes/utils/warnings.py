"""Warnings used by astartes"""


class ImperfectSplittingWarning(RuntimeWarning):
    """Used when a sampler cannot match requested splits."""

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class NormalizationWarning(RuntimeWarning):
    """Used when a requested split does not add to 1."""

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class NoMatchingScaffold(Warning):
    """
    Used when an RDKit molecule does not match any
    Bemis-Murcko scaffold and returns an empty string.
    """

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)
