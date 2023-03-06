from sklearn.cluster import DBSCAN as sk_DBSCAN

from astartes.samplers import AbstractSampler


class DBSCAN(AbstractSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _sample(self):
        """
        Implements the DBSCAN sampler to identify clusters.
        Density-Based Spatial Clustering of Applications with Noise finds
        core samples in regions of high density and expands clusters from them.
        This algorithm is good for data which contains clusters of similar density.
        """

        dbscan = sk_DBSCAN(
            eps=self.get_config("eps", 0.5),
            min_samples=self.get_config("min_samples", 2),
            metric=self.get_config("metric", "euclidean"),
            algorithm=self.get_config("algorithm", "auto"),
            leaf_size=self.get_config("leaf_size", 30),
            p=self.get_config("p", None),
        ).fit(self.X)
        self._samples_clusters = dbscan.labels_
