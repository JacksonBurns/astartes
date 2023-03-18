import unittest

import numpy as np

from astartes import train_test_split
from astartes.samplers import KMeans
from astartes.utils.warnings import ImperfectSplittingWarning


class Test_kmeans(unittest.TestCase):
    """
    Test the various functionalities of kmeans.
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

    def test_kmeans_sampling(self):
        """Use kmeans in the train_test_split and verify results."""
        with self.assertWarns(ImperfectSplittingWarning):
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
                test_size=0.75,
                train_size=0.25,
                sampler="kmeans",
                random_state=42,
                hopts={
                    "n_clusters": 2,
                },
            )
        # test that the known arrays equal the result from above
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_train,
                np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0]]),
            ),
            "Train X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_test,
                np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]),
            ),
            "Test X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_train,
                np.array([1, 2, 3]),
            ),
            "Train y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_test,
                np.array([4, 5]),
            ),
            "Test y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_train,
                np.array(["one", "two", "three"]),
            ),
            "Train labels incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_test,
                np.array(["four", "five"]),
            ),
            "Test labels incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                clusters_train,
                np.array([1, 1, 1]),
            ),
            "Train clusters incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                clusters_test,
                np.array([0, 0]),
            ),
            "Test clusters incorrect.",
        )

    def test_kmeans(self):
        """Directly instantiate and test KMeans."""
        kmeans_instance = KMeans(
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
            KMeans,
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
