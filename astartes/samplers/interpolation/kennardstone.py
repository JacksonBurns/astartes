import numpy as np
from scipy.spatial.distance import pdist, squareform

from astartes.samplers import AbstractSampler
from astartes.utils.exceptions import InvalidConfigurationError


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
        np.fill_diagonal(ks_distance, -np.inf)

        # get the row/col of maximum (greatest distance)
        max_idx = np.nanargmax(ks_distance)
        max_coords = np.unravel_index(max_idx, ks_distance.shape)

        # -np.inf these points to trick numpy later on
        ks_distance[max_coords[0], :] = -np.inf
        ks_distance[max_coords[1], :] = -np.inf

        # list of indices which have been selected, for use in the below loop
        already_selected = list()
        already_selected.append(max_coords[0])
        already_selected.append(max_coords[1])

        # iterate through the rest
        for _ in range(n_samples - 2):
            # find the next sample with the largest minimum distance to any sample already selected
            # get out the columns for the data already selected
            select_ks_distance = ks_distance[:, already_selected]
            # find which member of the selected data each of the unselected data is closest to
            min_distances_vals = np.nanmin(select_ks_distance, axis=1)
            # pick the largest of those values
            max_min_idx = np.nanargmax(min_distances_vals)
            ks_distance[max_min_idx, :] = -np.inf
            already_selected.append(max_min_idx)

        self._samples_idxs = np.array(already_selected, dtype=int)
