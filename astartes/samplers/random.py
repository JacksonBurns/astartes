from .sampler import Sampler

# https://github.com/yu9824/kennard_stone
from sklearn.model_selection import train_test_split


class Random(Sampler):

    def __init__(self, configs):
        self._ks_split(self)

    def _ks_split(self):
        _, _, samples_idxs, _ = train_test_split(
            self.X, [0]*len(self.X),
            train_size=1.0,
        )
        self._samples_idxs = samples_idxs

    def get_next_sample_idx(self):
        return self._res.pop(0)
