from astartes.samplers import AbstractSampler

from math import floor

from sklearn.cluster import KMeans as sk_KMeans


class KMeans(AbstractSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _sample(self):
        kmeanModel = sk_KMeans(
            n_clusters=self.get_config("n_clusters", floor(len(self.X) * 0.1) + 1),
            n_init=self.get_config("n_init", 10),
            random_state=self.get_config("random_state", None),
        ).fit(self.X)
        print(kmeanModel.labels_)
        # output list of two-tuples with index and cluster, or find a better way
        # sort in order of smallest to largest cluster, use that sort to sort the labels as well
        pass
