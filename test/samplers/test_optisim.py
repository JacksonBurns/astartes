import os
import sys
import unittest

import numpy as np

from astartes.samplers import OptiSim


class Test_optisim(unittest.TestCase):
    """
    Test the various functionalities of optisim.
    """

    @classmethod
    def setUpClass(self):
        """Convenience attributes for later tests."""
        self.X = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
            ]
        )
        self.y = np.array([1, 2, 3, 4, 5])
        self.labels = np.array(
            [
                "one",
                "two",
                "three",
                "four",
                "five",
            ]
        )

    def test_optisim(self):
        """Directly instantiate and test OptiSim"""
        kmeans_instance = OptiSim(
            self.X,
            self.y,
            self.labels,
            {
                "n_clusters": 2,
                "random_state": 42,
            },
        )
        self.assertIsInstance(
            kmeans_instance,
            OptiSim,
            "Failed instantiation.",
        )
        self.assertTrue(
            len(kmeans_instance.get_clusters()),
            "Clusters not set.",
        )
        self.assertTrue(
            len(kmeans_instance.get_sorted_cluster_counter()),
            "Sorted cluster Counter not found.",
        )
        self.assertTrue(
            len(kmeans_instance._samples_idxs),
            "Sample indices not set.",
        )


if __name__ == "__main__":
    unittest.main()
