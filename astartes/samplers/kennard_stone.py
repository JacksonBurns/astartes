import numpy as np
from numpy.linalg import matrix_rank
from scipy.spatial.distance import pdist, squareform

from sampler import Sampler


class KennardStone(Sampler):
    """
    Implements the algorithm outlined in
    Kennard, R. W., & Stone, L. A. (1969).
    Computer Aided Design of Experiments. Technometrics, 11(1), 137–148.
    https://doi.org/10.1080/00401706.1969.10490666

    """
    def __init__(self, X, y=None, distance_matrix=None, initial_point_id=None):
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
        return pdist(self.X)

    def get_sample(self):
        pass

    def get_sample_id(self):
        pass

    def get_batch_sample(self, n_samples):
        pass

    def get_batch_sample_idx(self, n_samples):
        pass