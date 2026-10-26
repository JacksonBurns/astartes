import os
import sys
import unittest

import numpy as np

from astartes import train_test_split
from astartes.samplers import OptiSim
from astartes.utils.warnings import ImperfectSplittingWarning


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

    def test_optisim_sampling(self):
        """Use optisim in the train_test_split and verify results."""
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
                sampler="optisim",
                random_state=42,
                hopts={
                    "n_clusters": 2,
                },
            )
        # test that the known arrays equal the result from above
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_train,
                np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                    ]
                ),
                "Train X incorrect.",
            ),
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_test,
                np.array(
                    [
                        [1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0],
                    ]
                ),
                "Test X incorrect.",
            ),
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_train,
                np.array([1, 2, 3]),
                "Train y incorrect.",
            ),
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_test,
                np.array([4, 5]),
                "Test y incorrect.",
            ),
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_train,
                np.array(["one", "two", "three"]),
                "Train labels incorrect.",
            ),
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_test,
                np.array(["four", "five"]),
                "Test labels incorrect.",
            ),
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                clusters_train,
                np.array([0, 0, 0]),
                "Train clusters incorrect.",
            ),
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                clusters_test,
                np.array([1, 1]),
                "Test clusters incorrect.",
            ),
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

    def test_get_dist_properties(self):
        """Verify that get_dist is symmetric and zero along diagonal (issue #187)."""
        sampler = OptiSim(self.X, configs={"n_clusters": 2})
        n_samples = len(self.X)
        for i in range(n_samples):
            self.assertEqual(
                sampler.get_dist(i, i),
                0.0,
                f"get_dist({i}, {i}) should be 0.0",
            )
            for j in range(n_samples):
                self.assertEqual(
                    sampler.get_dist(i, j),
                    sampler.get_dist(j, i),
                    f"get_dist({i}, {j}) != get_dist({j}, {i})",
                )

    def test_optisim_cluster_recovery(self):
        """Verify that OptiSim correctly recovers separated clusters (issue #187)."""
        from sklearn.metrics import adjusted_rand_score

        rng = np.random.default_rng(0)
        centres = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=float)
        true = np.repeat(np.arange(4), 25)
        X = centres[true] + rng.normal(scale=1.0, size=(100, 2))
        order = rng.permutation(100)
        X, true = X[order], true[order]

        for seed in range(5):
            sampler = OptiSim(X, configs={"n_clusters": 4, "random_state": seed})
            ari = adjusted_rand_score(true, sampler.get_clusters())
            self.assertAlmostEqual(
                ari,
                1.0,
                places=5,
                msg=f"OptiSim failed to recover separated clusters for seed {seed} (ARI={ari})",
            )


if __name__ == "__main__":
    unittest.main()
