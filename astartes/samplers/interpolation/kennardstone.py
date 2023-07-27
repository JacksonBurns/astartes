from astartes.samplers import AbstractSampler
from astartes.utils.fast_kennard_stone import fast_kennard_stone
from scipy.spatial.distance import pdist, squareform


class KennardStone(AbstractSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _sample(self):
        """
        Implements the Kennard-Stone algorithm
        """
        self._samples_idxs = fast_kennard_stone(
            squareform(
                pdist(
                    self.X,
                    self.get_config("metric", "euclidean"),
                )
            )
        )
