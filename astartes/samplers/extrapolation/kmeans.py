from collections import Counter
from math import floor

import numpy as np
from sklearn.cluster import KMeans as sk_KMeans

from astartes.samplers import AbstractSampler


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
