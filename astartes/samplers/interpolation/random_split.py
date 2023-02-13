from astartes.samplers import AbstractSampler
import numpy as np
import random


class Random(AbstractSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _sample(self):
        idx_list = list(range(len(self.X)))
        random.Random(self.get_config("random_state", None)).shuffle(idx_list)
        self._samples_idxs = np.array(idx_list, dtype=int)
