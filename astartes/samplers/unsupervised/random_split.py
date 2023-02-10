from astartes.samplers import AbstractSampler

import random


class Random(AbstractSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _sample(self):
        idx_list = list(range(len(self.X)))
        random.Random(self._configs.get("random_state", None)).shuffle(idx_list)
        self._samples_idxs = idx_list
