import os
import sys
import unittest

import numpy as np

from astartes import train_test_split
from astartes.samplers.extrapolation import DBSCAN


class Test_DBSCAN(unittest.TestCase):
    """
    Test the various functionalities of dbscan.
    """

    @classmethod
    def setUpClass(self):
        # create 5 clusters
        self.X = np.array(
            [
                [0.1, 0.2],
                [0, 0],
                [-0.1, -0.1],

                [1.1, 1.3],
                [1, 1],
                [0.9, 0.8],

                [-1.3, -1.1],
                [-1, -1],
                [-0.9, -0.8],

                [-1.2, 1.1],
                [-1, 1],
                [-0.9, 0.8],

                [1.1, -1.2],
                [1, -1],
                [0.9, -0.7],
            ]
        )

        self.y = np.array(
            [1, 1, 1,
             2, 2, 2,
             3, 3, 3,
             4, 4, 4,
             5, 5, 5,
            ]
        )

        self.labels = np.array(
            [
                "one", "one", "one",
                "two", "two", "two",
                "three", "three", "three",
                "four", "four", "four",
                "five", "five", "five",
            ]
        )

    def test_dbscan_sampling(self):
        """Use dbscan in the train_test_split and verify results."""
        (
                X_train,
                X_test,
                y_train,
                y_test,
                labels_train,
                labels_test,
                clusters_train,
                clusters_test,
            ) = train_test_split(
                self.X,
                self.y,
                labels=self.labels,
                test_size=0.2,
                train_size=0.8,
                sampler="dbscan",
                hopts={
                    "eps": 1,
                    "min_samples": 3,
                },
            )

        # test that the known arrays equal the result from above
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_train,
                np.array([
                    [1.1, 1.3],
                    [1, 1],
                    [0.9, 0.8],

                    [-1.3, -1.1],
                    [-1, -1],
                    [-0.9, -0.8],

                    [-1.2, 1.1],
                    [-1, 1],
                    [-0.9, 0.8],

                    [1.1, -1.2],
                    [1, -1],
                    [0.9, -0.7],
                ]),
            ),
            "Train X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_test,
                np.array([
                    [0.1, 0.2],
                    [0, 0],
                    [-0.1, -0.1],
                ]),
            ),
            "Test X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_train,
                np.array(
                    [2, 2, 2,
                     3, 3, 3,
                     4, 4, 4,
                     5, 5, 5,
                    ]
                ),
            ),
            "Train y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_test,
                np.array([1, 1, 1]),
            ),
            "Test y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_train,
                np.array(
                    ["two", "two", "two",
                     "three", "three", "three",
                     "four", "four", "four",
                     "five", "five", "five"
                     ]
                ),
            ),
            "Train labels incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_test,
                np.array(["one", "one", "one"]),
            ),
            "Test labels incorrect.",
        )
        print('clusters_train')
        print(clusters_train)
        self.assertIsNone(
            np.testing.assert_array_equal(
                clusters_train,
                np.array(
                    [1, 1, 1,
                     2, 2, 2,
                     3, 3, 3,
                     4, 4, 4,
                    ]
                )
            ),
            "Train clusters incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                clusters_test,
                np.array([0, 0, 0]),
            ),
            "Test clusters incorrect.",
        )

    def test_dbscan(self):
        """Directly instantiate and test DBSCAN."""
        dbscan_instance = DBSCAN(
            self.X,
            self.y,
            self.labels,
            {
                "eps": 1,
                "min_samples": 3,
            },
        )
        self.assertIsInstance(
            dbscan_instance,
            DBSCAN,
            "Failed instantiation.",
        )
        self.assertTrue(
            len(dbscan_instance.get_clusters()),
            "Clusters not set.",
        )
        self.assertTrue(
            len(dbscan_instance.get_sorted_cluster_counter()),
            "Sorted cluster Counter not found.",
        )
        self.assertTrue(
            len(dbscan_instance._samples_idxs),
            "Sample indices not set.",
        )


if __name__ == "__main__":
    unittest.main()
