"""Abstract Sampling class"""
from abc import ABC, abstractmethod

from astartes.utils.exceptions import (
    NotInitializedError,
    DatasetError,
)

from astartes.samplers import AbstractUnsupervisedSampler


class AbstractSupervisedSampler(AbstractUnsupervisedSampler):
    """
    Abstract Base Class for Supervised samplers.
    """

    @abstractmethod
    def __init__(self, configs):
        pass

    @abstractmethod
    def _get_next_sample_idx(self):
        """
        Get the idx of the next sample.
        """
        pass

    def populate(self, X, labels, y=None):
        """
        Load data in the instance.
        """
        self.X = X
        self.y = y
        self.labels = labels
        self.is_populated = True
        self.sample_count = 0
