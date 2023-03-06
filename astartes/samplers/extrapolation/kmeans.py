from math import floor

from sklearn.cluster import KMeans as sk_KMeans

from astartes.samplers import AbstractSampler


class KMeans(AbstractSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _sample(self):
        """Implements the K-Means sampler to identify clusters."""
        # use the sklearn kmeans model
        default_kwargs = {
            "n_clusters": self.get_config("n_clusters", floor(len(self.X) * 0.1) + 1),
            "init": self.get_config("init", "k-means++"),
            "n_init": self.get_config("n_init", 1),
            "max_iter": self.get_config("max_iter", 300),
            "tol": self.get_config("tol", 1e-4),
            "verbose": self.get_config("verbose", 0),
            "random_state": self.get_config("random_state", None),
            "copy_x": self.get_config("copy_x", True),
        }
        if self.get_config("algorithm", False):
            default_kwargs["algorithm"] = self.get_config("algorithm")

        kmeanModel = sk_KMeans(**default_kwargs).fit(self.X)
        self._samples_clusters = kmeanModel.labels_
