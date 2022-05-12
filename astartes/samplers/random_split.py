from astartes.samplers import Sampler

# https://github.com/yu9824/kennard_stone
from sklearn.model_selection import train_test_split


class Random(Sampler):

    def __init__(self, configs):
        self._split = False
        self._random_state = configs.get('random_state', None)
        self._shuffle = configs.get('shuffle', True)
        self._samples_idxs = []

    def _rand_split(self):
        _, _, samples_idxs, spare_idx = train_test_split(
            self.X, list(range(len(self.X))),
            train_size=len(self.X)-1,
            random_state=self._random_state,
            shuffle=self._shuffle,
        )
        self._samples_idxs = samples_idxs + spare_idx

    def _get_next_sample_idx(self):
        if self._split:
            return self._samples_idxs.pop(0)
        else:
            self._rand_split()
            self._split = True
            return self._get_next_sample_idx()
