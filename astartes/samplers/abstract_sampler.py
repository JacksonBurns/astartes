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
        self._sorted_cluster_counter = {}
        self._current_sample_idx = 0
        self._sample()
        if len(self._samples_clusters):
            self._set_sorted_cluster_counter()

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
        out = self._samples_idxs[
            self._current_sample_idx : self._current_sample_idx + n_samples
        ]
        self._current_sample_idx += n_samples
        return out

    def get_sorted_cluster_counter(self):
        """Getter for self._sorted_cluster_counter, used by main

        Returns:
            dict: The dictionary of clusters and lengths, sorted by length.
        """
        return self._sorted_cluster_counter

    def get_clusters(self):
        """Getter for the cluster labels.

        Returns:
            np.array: Cluster labels.
        """
        return self._samples_clusters

    def _set_sorted_cluster_counter(self):
        """Sets self._sorted_cluster_counter with a dict containing cluster_id:
        #_elts sorted by #_elts, ascending"""
        # need to sort labels and clusters in order of smallest cluster to largest
        # start by counting the number of elements in each cluster
        cluster_counter = Counter(self._samples_clusters)

        # create an integer array that stores the indices as they are now, which we
        # will sort alongside the labels based on the number of elements in each
        # cluster (effectively tracking the mapping between unsorted and sorted)
        samples_idxs = np.array(range(len(self.X)), dtype=int)

        sorted_idxs = []
        sorted_cluster_counter = {}
        for label, sample_idx in sorted(
            zip(
                self._samples_clusters, samples_idxs
            ),  # iterate indices and labels simultaneously
            key=lambda pair: cluster_counter[
                pair[0]
            ],  # use the number of elements in the cluster for sorting
        ):
            sorted_idxs.append(sample_idx)
            if label not in sorted_cluster_counter:
                sorted_cluster_counter[label] = 1
            else:
                sorted_cluster_counter[label] += 1

        # can't use np.argsort because it does not allow for a custom comparison key
        # and will instead sort by the value of the cluster labels (wrong!)
        self._samples_idxs = np.array(sorted_idxs, dtype=int)
        self._sorted_cluster_counter = sorted_cluster_counter

    def get_semi_sorted_cluster_counter(self, max_shufflable_size):
        """Similar to sorted cluster counter, except that cluster with fewer elements
        than max_shufflable_size will have their order shuffled."""
        # start with the sorted cluster counter from above
        cc = self.get_sorted_cluster_counter()
        # create a list of cluster labels where the cluster is not longer than
        # than max_shufflable_size
        small_clusters = [
            cluster_label
            for cluster_label, length in cc.items()
            if length <= max_shufflable_size
        ]
        # the remaining clusters go here
        large_clusters = [i for i in cc.keys() if i not in small_clusters]
        # shuffle the small clusters according to the random state
        rng = np.random.default_rng(seed=self.get_config("random_state"))
        rng.shuffle(small_clusters)
        # recombine the clusters
        all_clusters = small_clusters + large_clusters
        # track the indices of the cluster membership that we can easily
        # keep track of the new order after the clusters are sorted
        cc_with_idxs = {
            cluster_label: np.where(self._samples_clusters == cluster_label)[0]
            for cluster_label, length in cc.items()
        }
        # loop through all clusters, update the semi-sorted dictionary and
        # concatenate the lists of indices to arrive at a final sorted list
        semi_sorted_cc = {}
        new_samples_idxs = np.array([], dtype=int)
        for cluster_label in all_clusters:
            semi_sorted_cc[cluster_label] = cc[cluster_label]
            new_samples_idxs = np.hstack(
                (new_samples_idxs, cc_with_idxs[cluster_label])
            )
        self._samples_idxs = new_samples_idxs
        return semi_sorted_cc
