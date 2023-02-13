from astartes.samplers import AbstractSampler

from math import floor
from collections import Counter

from sklearn.cluster import KMeans as sk_KMeans
import numpy as np


class KMeans(AbstractSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _sample(self):
        """Implements the K-Means sampler to identify clusters."""
        # use the sklearn kmeans model
        kmeanModel = sk_KMeans(
            n_clusters=self.get_config("n_clusters", floor(len(self.X) * 0.1) + 1),
            n_init=self.get_config("n_init", 1),
            random_state=self.get_config("random_state", None),
        ).fit(self.X)
        self._samples_clusters = kmeanModel.labels_

        # need to sort labels and clusters in order of smallest cluster to largest
        # start by counting the number of elements in each cluster
        cluster_counter = Counter(kmeanModel.labels_)

        # create an integer array that stores the indices as they are now, which we
        # will sort alongside the labels based on the number of elements in each
        # cluster (effectively tracking the mapping between unsorted and sorted)
        samples_idxs = np.array(range(len(self.X)), dtype=int)

        # workhorse - this and some lines above/below could be made into a private
        # method in AbstractSampler
        sorted_idxs = []
        sorted_cluster_counter = {}
        for label, sample_idx in sorted(
            zip(
                kmeanModel.labels_, samples_idxs
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
