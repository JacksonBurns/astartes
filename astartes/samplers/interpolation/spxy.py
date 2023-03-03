"""
Implements the Sample set Partitioning based on join X-Y distances
algorithm as originally described by Saldanha and coworkers in
"A method for calibration and validation subset partitioning"
doi:10.1016/j.talanta.2005.03.025

This implementation has been validated against their original source
code implementation, which can be found in the paper linked above.
The corresponding unit tests reflect the expected output from
the original implemenation
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

from astartes.samplers import AbstractSampler
from astartes.utils.exceptions import InvalidConfigurationError


class SPXY(AbstractSampler):
    def __init__(self, *args):
        super().__init__(*args)

        if self.y is None:
            raise InvalidConfigurationError(
                "SPXY sampler requires both X and y arrays. Provide y or switch to kennard_stone."
            )

    def _sample(self):
        """
        Implements the SPXY algorithm as described by Saldahna et al.
        """
        distance_metric = self.get_config("metric", "euclidean")

        y_2d = self.y[:, np.newaxis]

        y_pdist = pdist(y_2d, metric=distance_metric)
        y_pdist = np.divide(y_pdist, np.amax(y_pdist))

        X_dist = pdist(self.X, metric=distance_metric)
        X_dist = np.divide(X_dist, np.amax(X_dist))

        # sum the distances as per eq. 3 of Saldahna, set diagonal to nan
        spxy_distance = squareform(y_pdist + X_dist)
        np.fill_diagonal(spxy_distance, np.nan)

        print(spxy_distance, np.argmin(spxy_distance, keepdims=True))

        self._samples_idxs = np.array([], dtype=int)

    def _pdist_lookup(self, i, j):
        """Convert matrix i,j to flattened index"""
        return len(self.X) * i + j - ((i + 2) * (i + 1)) // 2
