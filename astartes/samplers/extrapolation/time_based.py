from datetime import date, datetime

import numpy as np

from astartes.samplers import AbstractSampler


class TimeBased(AbstractSampler):
    def _before_sample(self):
        # verify that the user provided time as the labels (i.e. args[2])
        if self.labels is None:
            msg = "Time based splitting requires the input labels to be a date or datetime object"
            raise ValueError(msg)

        # verify that labels (i.e. args[2]) contains the expected data type
        elif not all(isinstance(i, date) for i in self.labels) and not all(isinstance(i, datetime) for i in self.labels):
            msg = "Time based splitting requires the input labels to be an iterable of date or datetime objects"
            raise TypeError(msg)

    def _sample(self):
        """
        Implements the time-based sampler to create an extrapolation split.
        This places new data points in the testing set and older data points in the training set.
        """
        data = [(time, idx) for idx, time in zip(np.arange(len(self.labels)), self.labels)]
        sorted_list = sorted(data, reverse=False)

        self._samples_idxs = np.array([idx for time, idx in sorted_list], dtype=int)
