"""
Implements the Sample set Partitioning based on join X-Y distances
algorithm as originally described by Saldanha and coworkers in
"A method for calibration and validation subset partitioning"
doi:10.1016/j.talanta.2005.03.025

This implementation has been validated against their original source
code implementation, which can be found in the paper linked above.
The corresponding unit tests reflect the expected output from
the original implemenation. The breaking of ties is different
compared to the original, but this is ultimately a minor and
likely inconsequential difference.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

from astartes.samplers import AbstractSampler
from astartes.utils.exceptions import InvalidConfigurationError


class SPXY(AbstractSampler):
    def __init__(self, *args):
        if args[1] is None:
            raise InvalidConfigurationError(
                "SPXY sampler requires both X and y arrays. Provide y or switch to kennard_stone."
            )
        super().__init__(*args)

    def _sample(self):
        """
        Implements the SPXY algorithm as described by Saldahna et al.
        """
        n_samples = len(self.X)

        distance_metric = self.get_config("metric", "euclidean")

        y_2d = self.y[:, np.newaxis]

        y_pdist = pdist(y_2d, metric=distance_metric)
        y_pdist = np.divide(y_pdist, np.amax(y_pdist))

        X_dist = pdist(self.X, metric=distance_metric)
        X_dist = np.divide(X_dist, np.amax(X_dist))

        # sum the distances as per eq. 3 of Saldahna, set diagonal to nan
        spxy_distance = squareform(y_pdist + X_dist)
        np.fill_diagonal(spxy_distance, -np.inf)

        # get the row/col of maximum (greatest distance)
        max_idx = np.nanargmax(spxy_distance)
        max_coords = np.unravel_index(max_idx, spxy_distance.shape)

        # -np.inf these points to trick numpy later on
        spxy_distance[max_coords[0], :] = -np.inf
        spxy_distance[max_coords[1], :] = -np.inf

        # list of indices which have been selected, for use in the below loop
        alread_selected = list()
        alread_selected.append(max_coords[0])
        alread_selected.append(max_coords[1])

        # iterate through the rest
        for _ in range(n_samples - 2):
            # find the next sample with the largest minimum distance to any sample already selected
            # get out the columns for the data already selected
            select_spxy_distance = spxy_distance[:, alread_selected]
            # find which member of the selected data each of the unselected data is closest to
            min_distances_vals = np.nanmin(select_spxy_distance, axis=1)
            # pick the largest of those values
            max_min_idx = np.nanargmax(min_distances_vals)
            spxy_distance[max_min_idx, :] = -np.inf
            alread_selected.append(max_min_idx)

        self._samples_idxs = np.array(alread_selected, dtype=int)
