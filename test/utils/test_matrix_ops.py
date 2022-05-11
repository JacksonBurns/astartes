import unittest

import numpy as np
from numpy.random import rand
from scipy.spatial.distance import squareform

from astartes.utils import matrix_ops


class TestMatrixOps(unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        n_sample_dims = 50
        A = rand(self.n_samples, n_sample_dims)
        self.symmetric_square_matrix = np.matmul(A, np.transpose(A))
        for i in range(self.n_samples):
            self.symmetric_square_matrix[i, i] = 0.
        self.condensed_matrix = squareform(self.symmetric_square_matrix)

    def test_square_to_condensed(self):
        for row in range(self.n_samples):
            for col in range(self.n_samples):
                if row == col:
                    continue
                condensed_id = matrix_ops.square_to_condensed(
                    row,
                    col,
                    self.n_samples)
                self.assertEqual(self.symmetric_square_matrix[row, col],
                                 self.condensed_matrix[condensed_id],
                                 'Expected square matrix element to be equal '
                                 'to the corresponding condensed entry.')

    def test_condensed_to_square(self):
        for condensed_id, condensed_elem in enumerate(self.condensed_matrix):
            row, col = matrix_ops.condensed_to_square(condensed_id,
                                                      self.n_samples)
            self.assertEqual(condensed_elem, self.symmetric_square_matrix[row,
                                                                          col],
                             'Expected square matrix element to be equal '
                             'to the corresponding condensed entry.')
