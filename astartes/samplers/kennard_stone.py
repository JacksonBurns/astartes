import numpy as np
from numpy.linalg import matrix_rank
from scipy.spatial.distance import pdist, squareform

from ..utils import matrix_ops
from .sampler import Sampler


class KennardStone(Sampler):
    """
    Implements the algorithm outlined in
    Kennard, R. W., & Stone, L. A. (1969).
    Computer Aided Design of Experiments. Technometrics, 11(1), 137–148.
    https://doi.org/10.1080/00401706.1969.10490666

    """

    def __init__(self, X, y=None, distance_matrix=None, initial_point_id=None):
        """

        Args:
            X:
            y:
            distance_matrix:
            initial_point_id:
        """
        self.X = X
        self.y = y
        if distance_matrix is None:
            distance_matrix = self._get_l2_distance_matrix()
        else:
            distance_matrix = np.array(distance_matrix)
            if matrix_rank(distance_matrix) > 1:
                distance_matrix = squareform(distance_matrix, checks=True)
        self.distance_matrix = distance_matrix
        self.sample_idx = []
        if initial_point_id is not None:
            self.sample_idx.append(initial_point_id)

    def _get_l2_distance_matrix(self):
        """

        Returns:

        """
        return pdist(self.X)

    def _is_init(self):
        """

        Returns:
            (bool): True if sampler is initialized.

        """
        return len(self.sample_idx) > 0

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

    def get_sample(self):
        """

        Returns:

        """
        return self.X[self.get_sample_id(), :]

    def get_sample_id(self):
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

    def get_batch_sample(self, n_samples):
        return self.X[self.get_batch_sample_idx(n_samples), :]

    def get_batch_sample_idx(self, n_samples):
        return [self.get_sample_id() for _ in range(n_samples)]

    def get_initial_point_idx(self):
        return self.sample_idx[0]

    def get_initial_point(self):
        return self.X[self.get_initial_point_idx(), :]
