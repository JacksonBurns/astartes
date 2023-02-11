"""Abstract Sampling class"""
from abc import ABC, abstractmethod
from collections import Counter

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
        self._samples_clusters = []
        self._current_sample_idx = 0
        self._sample()

    @abstractmethod
    def _sample(self):
        """This method should set self._samples_idxs (and self._samples_clusters if usng extrapolative method)"""
        pass

    def get_config(self, key, default=None):
        return self._configs.get(key, default)

    def get_sample_idxs(self, n_samples):
        """
        Get idxs of samples.
        """
        out = self._samples_idxs[
            self._current_sample_idx : self._current_sample_idx + n_samples
        ]
        self._current_sample_idx += n_samples
        return out

    def get_cluster_counter(self):
        return Counter(self._samples_clusters)

    def get_clusters(self):
        return self._samples_clusters
