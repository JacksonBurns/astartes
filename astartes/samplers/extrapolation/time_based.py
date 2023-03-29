from datetime import date, datetime

import numpy as np

from astartes.samplers import AbstractSampler


class TimeBased(AbstractSampler):
    def __init__(self, *args):
        # verify that the user provided time as the labels (i.e. args[2])
        if args[2] is None:
            msg = "Time based splitting requires the input labels to be a date or datetime object"
            raise ValueError(msg)

        # verify that labels (i.e. args[2]) contains the expected data type
        elif not isinstance(args[2][0], date) and not isinstance(args[2][0], datetime):
            msg = "Time based splitting requires the input labels to be an iterable of date or datetime objects"
            raise TypeError(msg)

        super().__init__(*args)

    def _sample(self):
        """
        Implements the time-based sampler to create an extrapolation split.
        This places new data points in the testing set and older data points in the training set.
        """
        data = [(time, idx) for idx, time in zip(np.arange(len(self.labels)), self.labels)]
        sorted_list = sorted(data, reverse=False)

        self._samples_idxs = np.array([idx for time, idx in sorted_list], dtype=int)
