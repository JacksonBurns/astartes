from astartes.samplers import Sampler

# https://github.com/yu9824/kennard_stone
from kennard_stone import train_test_split


class KennardStone(Sampler):

    def __init__(self, configs):
        self._split = False
        self._samples_idxs = []

    def _ks_split(self):
        """
        Uses another implementation of KS split to get the order in which
        the data would be sampled.

        SKLearn does not allow sampling of all the data into the training
        set, so we ask for all but one of the points, and then put that
        index into the list at the end to circumvent this.
        """
        _, _, samples_idxs, spare_idx = train_test_split(
            self.X, list(range(len(self.X))),
            train_size=len(self.X)-1,
        )
        self._samples_idxs = samples_idxs + spare_idx

    def _get_next_sample_idx(self):
        if self._split:
            return self._samples_idxs.pop(0)
        else:
            self._ks_split()
            self._split = True
            return self._get_next_sample_idx()
