"""
This sampler partitions the data based on the regression target y. It first sorts the
data by y value and then constructs the training set to have either the smallest (largest)
y values, the validation set to have the next smallest (largest) set of y values, and the
testing set to have the largest (smallest) y values.
"""

import numpy as np

from astartes.samplers import AbstractSampler


class TargetProperty(AbstractSampler):
    def _sample(self):
        """
        Implements the target property sampler to create an extrapolation split.
        """
        data = [(y, idx) for y, idx in zip(self.y, np.arange(len(self.y)))]

        # by default, the smallest property values are placed in the training set
        sorted_list = sorted(data, reverse=self.get_config("descending", False))

        self._samples_idxs = np.array([idx for time, idx in sorted_list], dtype=int)
