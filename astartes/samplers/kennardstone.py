from .sampler import Sampler
from ..utils import matrix_ops
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import matrix_rank
import numpy as np
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


class KennardStone(Sampler):
    """
    Implements the algorithm outlined in
    Kennard, R. W., & Stone, L. A. (1969).
    Computer Aided Design of Experiments. Technometrics, 11(1), 137–148.
    https://doi.org/10.1080/00401706.1969.10490666
    """

    def __init__(self, configs):
        """
        Args:
            X:
            y:
            distance_matrix:
            initial_point_id:
        """
        self._distance_matrix = configs.get('distance_matrix', None)
        self._initial_point_id = configs.get('initial_point_id', None)
        return self

    def _get_heuristic_init_points(self):
        max_dist = -np.inf
        idx1, idx2 = -1, -1
        n_samples = self.X.shape[0]
        for row in range(n_samples-1):
            for col in range(row+1, n_samples):
                dist = self.distance_matrix[matrix_ops.square_to_condensed(
                    row,
                    col,
                    n_samples)]
                if dist > max_dist:
                    max_dist = dist
                    idx1, idx2 = row, col
        return idx1, idx2

    def _get_next_sample_idx(self):
        if self._distance_matrix is None:
            distance_matrix = pdist(self.X)
        else:
            distance_matrix = np.array(distance_matrix)
            if matrix_rank(distance_matrix) > 1:
                distance_matrix = squareform(distance_matrix, checks=True)

        self._sample_idx = []
        if self._initial_point_id is not None:
            self._sample_idx.append(self._initial_point_id)

        if not self._is_init():
            idx1, idx2 = self._get_heuristic_init_points()
            self.sample_idx.append(idx1)
            self.sample_idx.append(idx2)
        else:
            n_samples = self.X.shape[0]
            candidate_idx = set(range(n_samples)) - set(self.sample_idx)
            max_distance_to_closest_sample = -np.inf
            furthest_candidate_idx = -1
            for id_ in candidate_idx:
                min_distance_to_samples = min(
                    [self.distance_matrix[matrix_ops.square_to_condensed(
                        id_,
                        col,
                        n_samples)]
                     for col in self.sample_idx])
                if min_distance_to_samples > max_distance_to_closest_sample:
                    max_distance_to_closest_sample = min_distance_to_samples
                    furthest_candidate_idx = id_
            self.sample_idx.append(furthest_candidate_idx)
        return self.sample_idx[-1]
