import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split

from astartes.samplers import AbstractSampler


class Random(AbstractSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _sample(self):
        """Passthrough to sklearn train_test_split"""
        idx_list = list(range(len(self.X)))
        train_indices, test_indices = sk_train_test_split(
            idx_list,
            train_size=len(idx_list) - 1,
            random_state=self.get_config("random_state", None),
            shuffle=self.get_config("shuffle", True),
        )
        self._samples_idxs = np.array(train_indices + test_indices, dtype=int)
