"""Abstract Sampling class"""
from abc import ABC, abstractmethod

import numpy as np


class AbstractSampler(ABC):
    """
    Abstract Base Class for all samplers.
    """

    def __init__(self, X, y, labels, configs):
        """Copies X, y, labels, and configs into class attributes and then calls sampler."""
        # required in all splitting methods
        self.X = X

        # optional in all splitting methods
        self.y = y
        self.labels = labels
        self._configs = configs

        # this must be set by _sample
        self._samples_idxs = np.array([], dtype=int)

        # these must also be set if using a clustering algorithm
        self._samples_clusters = np.array([], dtype=int)
        self._sorted_cluster_counter = {}

        # internal machinery
        self._current_sample_idx = 0
        self._sample()

    @abstractmethod
    def _sample(self):
        """
        This method should: (arrays should be np.ndarray)
         - set self._samples_idxs with the order in which the algorithm dictates drawing samples
        and if using clustering:
         - set self._samples_clusters with the labels produced by clustering
         - set self._sorted_cluster_counter with a dict containing cluter_id: #_elts sorted by #_elts, ascending
        """

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

    def get_sorted_cluster_counter(self):
        return self._sorted_cluster_counter

    def get_clusters(self):
        return self._samples_clusters
