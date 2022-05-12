"""Abstract Sampling class"""
from abc import ABC, abstractmethod

from astartes.exceptions import (
    NotInitializedError,
    DatasetError,
)


class Sampler(ABC):
    """
    Abstract Base Class for samplers.
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

    def populate(self, X, y=None):
        """
        Load data in the instance.
        """
        self.X = X
        self.y = y
        self.is_populated = True
        self.sample_count = 0

    def get_samples(self, n_samples):
        """
        Get samples.
        """
        self._verify_call(n_samples)
        return [self.X[i] for i in self.get_sample_idxs(n_samples)]

    def get_sample_idxs(self, n_samples):
        """
        Get idxs of samples.
        """
        self._verify_call(n_samples)
        return [self._get_next_sample_idx() for _ in range(n_samples)]

    def _verify_call(self, n_samples):
        if not self.is_populated:
            raise NotInitializedError(
                'Populate sampler instance with data to get samples'
            )
        if self.sample_count > len(self.X) or self.sample_count + n_samples > len(self.X):
            raise DatasetError(
                'Dataset exhausted.'
            )
        self.sample_count += n_samples
        return
