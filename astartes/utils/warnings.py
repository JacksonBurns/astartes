"""Warnings used by astartes"""


class ImperfectSplittingWarning(RuntimeWarning):
    """Used when a sampler cannot match requested splits."""

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)
