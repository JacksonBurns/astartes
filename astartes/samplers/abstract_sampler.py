"""Abstract Sampling class"""
from abc import ABC, abstractmethod

from astartes.utils.exceptions import (
    DatasetError,
)


class AbstractSampler(ABC):
    """
    Abstract Base Class for all samplers.
    """

    def __init__(self, X, y, labels, configs):
        """Copies X, y, labels, and configs into class attributes and then calls sampler."""
        self.X = X
        self.y = y
        self.labels = labels
        self._configs = configs
        self._samples_idxs = []
        self._sample()

    @abstractmethod
    def _sample(self):
        """This method should set self._samples_idxs"""
        pass

    def get_config(self, key, default=None):
        return self._configs.get(key, default)

    def get_sample_idxs(self, n_samples):
        """
        Get idxs of samples.
        """
        return [self._samples_idxs.pop(0) for _ in range(n_samples)]
