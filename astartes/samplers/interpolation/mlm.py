from typing import overload

import numpy as np

from astartes.samplers.interpolation import KennardStone


class MLM(KennardStone):
    # could be convenient to know size of train and test during init...
    @overload
    def get_sample_idxs(self, n_samples):
        """Overload the KennardStone method to permute 10% of samples from train

        Args:
            n_samples (int): Number of samples to retrieve.

        Returns:
            np.array: The selected indices
        """
        if self._current_sample_idx == 0:  # permute indexes on the first call
            train_idxs = self._samples_idxs[
                self._current_sample_idx : self._current_sample_idx + n_samples
            ]
            other_idxs = self._samples_idxs[self._current_sample_idx + n_samples : -1]

            # set RNG
            rng = np.random.default_rng(seed=self.get_config("random_state"))
            n_to_permute = np.floor(0.1 * len(train_idxs))
            train_permute_idxs = rng.choice(train_idxs, n_to_permute)
            remaining_train_idxs = [
                i for i in train_idxs if i not in train_permute_idxs
            ]
            other_permute_idxs = rng.choice(other_idxs, n_to_permute)
            remaining_other_idxs = [
                i for i in other_idxs if i not in other_permute_idxs
            ]
            # reassamble the indexes
            self._samples_idxs = np.hstack(
                (
                    remaining_train_idxs,
                    other_permute_idxs,
                    remaining_other_idxs,
                    train_permute_idxs,
                )
            )
        return super().get_sample_idxs(n_samples)
