"""Abstract Sampling class"""

from abc import ABC, abstractmethod
from collections import Counter

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

        # this must be set by _sample for interpolation
        self._samples_idxs = np.array([], dtype=int)

        # this must be set if using a clustering algorithm
        self._samples_clusters = np.array([], dtype=int)

        # internal machinery
        self._current_sample_idx = 0
        self._before_sample()
        self._sample()
        self._after_sample()

    def _before_sample(self):
        """This method should perform any data validation, manipulation, etc. required before proceeding to _sample

        Returns:
            None: Returns nothing, raises an Exception if something is wrong.
        """

    def _after_sample(self):
        """This method should perform any checks, mutations, etc. required after _sample is completed.

        Returns:
            None: Returns nothing, raises an Exception if something is wrong.
        """

    @abstractmethod
    def _sample(self):
        """
        This method should: (arrays should be np.ndarray)
         - set self._samples_idxs with the order in which the algorithm dictates drawing samples
        and if using clustering:
         - set self._samples_clusters with the labels produced by clustering
        """

    def get_config(self, key, default=None):
        """Getter to sampler._configs

        Args:
            key (str): String parameter for the sampler.
            default (any, optional): Default to return if key not present. Defaults to None.

        Returns:
            any: value at provided key, or else default.
        """
        return self._configs.get(key, default)

    def get_sample_idxs(self, n_samples):
        """
        Get idxs of samples.
        """
        out = self._samples_idxs[self._current_sample_idx : self._current_sample_idx + n_samples]
        self._current_sample_idx += n_samples
        return out

    def get_clusters(self):
        """Getter for the cluster labels.

        Returns:
            np.array: Cluster labels.
        """
        return self._samples_clusters

    def get_sorted_cluster_counter(self, max_shufflable_size=None):
        """
        Return a dict containing cluster_id: number of members sorted by number
        of members, ascending

        if max_shufflable_size is not None, clusters below the passed size will be
        shuffled into a new order according to random_state in hopts
        """
        # dictionary of cluster label: length (number of members)
        cluster_counter = Counter(self._samples_clusters)

        # same, sorted by number of members ascending
        ordered_cluster_counter = dict(sorted(cluster_counter.items(), key=lambda i: i[1]))

        # if using max_shufflable_size, shuffle the small clusters in this dictionary
        if max_shufflable_size is not None:
            # get a list of the small clusters
            small_clusters = [cluster_label for cluster_label, length in ordered_cluster_counter.items() if length <= max_shufflable_size]

            # the remaining clusters go here
            large_clusters = [i for i in ordered_cluster_counter.keys() if i not in small_clusters]

            # shuffle the small clusters according to the random state
            rng = np.random.default_rng(seed=self.get_config("random_state"))
            rng.shuffle(small_clusters)

            # recombine the clusters, rebuild the ordered dictionary
            all_clusters = small_clusters + large_clusters
            ordered_cluster_counter = {label: cluster_counter[label] for label in all_clusters}

        # make a new dictionary that maps cluster label: indexes of members
        ordered_cluster_counter_with_idxs = {
            cluster_label: np.where(self._samples_clusters == cluster_label)[0] for cluster_label, length in ordered_cluster_counter.items()
        }

        # put all the indices in order
        self._samples_idxs = np.hstack(tuple(ordered_cluster_counter_with_idxs.values()))

        return ordered_cluster_counter
