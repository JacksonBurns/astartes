import numpy as np
from scipy.spatial.distance import pdist, squareform

from astartes.samplers import AbstractSampler


class KennardStone(AbstractSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _sample(self):
        """
        Implements the Kennard-Stone algorithm
        """
        n_samples = len(self.X)

        distance_metric = self.get_config("metric", "euclidean")

        X_dist = pdist(self.X, metric=distance_metric)

        ks_distance = squareform(X_dist)
        # when searching for max distance, disregard self
        np.fill_diagonal(ks_distance, -np.inf)

        # delete from distance as we go, keep this array to keep track of idxs
        ks_idxs = np.arange(n_samples)

        # get the row/col of maximum (greatest distance)
        max_idx = np.nanargmax(ks_distance)
        max_coords = np.unravel_index(max_idx, ks_distance.shape)

        # delete these rows so we don't select them again
        ks_distance = np.delete(
            ks_distance,
            max_coords,
            axis=0,
        )
        # delete from the index list to track deletions in distance matrix
        ks_idxs = np.delete(ks_idxs, max_coords)

        # list of indices which have been selected, for use in the below loop
        already_selected = list(max_coords)
        min_distances = np.min(ks_distance[:, already_selected], axis=1)
        # print(min_distances)
        for _ in range(n_samples - 2):
            # print("already have", self.X[already_selected])
            # find the next sample with the largest minimum distance to any sample already selected
            max_min_idx = np.argmax(min_distances)
            # add to the selected, remove from index tracker and distance matrix
            ks_distance = np.delete(ks_distance, max_min_idx, axis=0)
            already_selected.append(ks_idxs[max_min_idx])
            ks_idxs = np.delete(ks_idxs, max_min_idx)
            # update the minimum distances array
            # delete the sample we just selected
            min_distances = np.delete(min_distances, max_min_idx)
            # get minimum distance of unselected samples to that sample only
            new_distances = np.min(ks_distance[:, [already_selected[-1]]], axis=1)
            min_distances = np.minimum(min_distances, new_distances)

        self._samples_idxs = np.array(already_selected, dtype=int)
