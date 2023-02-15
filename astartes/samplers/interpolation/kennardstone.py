# https://github.com/yu9824/kennard_stone
import numpy as np
from kennard_stone import train_test_split as ks_train_test_split

from astartes.samplers import AbstractSampler


class KennardStone(AbstractSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _sample(self):
        """
        Uses another implementation of KS split to get the order in which
        the data would be sampled.

        SKLearn does not allow sampling of all the data into the training
        set, so we ask for all but one of the points, and then put that
        index into the list at the end to circumvent this.
        """
        _, _, samples_idxs, spare_idx = ks_train_test_split(
            self.X,
            list(range(len(self.X))),
            train_size=len(self.X) - 1,
        )
        self._samples_idxs = np.array(samples_idxs + spare_idx, dtype=int)
