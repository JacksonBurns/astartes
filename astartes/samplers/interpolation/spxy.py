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
from astartes.utils.fast_kennard_stone import fast_kennard_stone


class SPXY(AbstractSampler):
    def _before_sample(self):
        if self.y is None:
            raise InvalidConfigurationError("SPXY sampler requires both X and y arrays. Provide y or switch to kennard_stone.")

    def _sample(self):
        """
        Implements the SPXY algorithm as described by Saldahna et al.
        """
        _default = self.get_config("distance_metric", False) or self.get_config("metric", "euclidean")
        distance_metric_X = self.get_config("metric", _default)
        distance_metric_y = self.get_config("metric", _default)

        y_2d = self.y[:, np.newaxis]

        y_pdist = pdist(y_2d, metric=distance_metric_X)
        y_pdist = np.divide(y_pdist, np.amax(y_pdist))

        X_dist = pdist(self.X, metric=distance_metric_y)
        X_dist = np.divide(X_dist, np.amax(X_dist))

        # sum the distances as per eq. 3 of Saldahna, set diagonal to nan
        spxy_distance = squareform(y_pdist + X_dist)

        self._samples_idxs = fast_kennard_stone(spxy_distance)
