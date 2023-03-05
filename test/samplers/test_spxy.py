import os
import sys
import unittest

import numpy as np

from astartes import train_test_split
from astartes.samplers import SPXY
from astartes.utils.exceptions import InvalidConfigurationError
from astartes.utils.warnings import ImperfectSplittingWarning


class Test_SPXY(unittest.TestCase):
    """
    Test the various functionalities of SPXY.
    """

    @classmethod
    def setUpClass(self):
        """Convenience attributes for later tests."""
        self.X = np.array(
            [
                [4, 1, 9, 5, 5, 7],
                [10, 9, 3, 3, 8, 2],
                [8, 7, 2, 7, 2, 1],
                [6, 8, 2, 2, 6, 10],
                [2, 1, 4, 3, 6, 10],
                [2, 10, 6, 4, 1, 9],
            ]
        )
        self.y = np.array([4, 1, 7, 5, 2, 5])
        self.labels = np.array(
            [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
            ]
        )

    def test_missing_y(self):
        """SPXY requires a y array and should complain when one is not provided."""
        with self.assertRaises(InvalidConfigurationError):
            train_test_split(
                self.X,
                y=None,
                sampler="spxy",
            )

    def test_spxy_sampling(self):
        """Use spxy in the train_test_split and verify results."""
        with self.assertWarns(ImperfectSplittingWarning):
            (
                X_train,
                X_test,
                y_train,
                y_test,
                labels_train,
                labels_test,
            ) = train_test_split(
                self.X,
                self.y,
                labels=self.labels,
                test_size=0.3,
                train_size=0.7,
                sampler="spxy",
                hopts={
                    "metric": "euclidean",
                },
            )
        # test that the known arrays equal the result from above
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_train,
                np.array(
                    [
                        [8, 7, 2, 7, 2, 1],
                        [2, 1, 4, 3, 6, 10],
                        [10, 9, 3, 3, 8, 2],
                        [2, 10, 6, 4, 1, 9],
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
                        [4, 1, 9, 5, 5, 7],
                        [6, 8, 2, 2, 6, 10],
                    ]
                ),
                "Test X incorrect.",
            ),
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_train,
                np.array([7, 2, 1, 5]),
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
                np.array(["three", "five", "two", "six"]),
                "Train labels incorrect.",
            ),
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_test,
                np.array(["one", "four"]),
                "Test labels incorrect.",
            ),
        )

    def test_spxy(self):
        """Directly instantiate and test SPXY"""
        spxy_instance = SPXY(
            self.X,
            self.y,
            self.labels,
            {},
        )
        self.assertIsInstance(
            spxy_instance,
            SPXY,
            "Failed instantiation.",
        )
        self.assertFalse(
            len(spxy_instance.get_clusters()),
            "Clusters should not have been set.",
        )
        self.assertFalse(
            len(spxy_instance.get_sorted_cluster_counter()),
            "Sorted cluster Counter should not be found.",
        )
        self.assertTrue(
            len(spxy_instance._samples_idxs),
            "Sample indices not set.",
        )


if __name__ == "__main__":
    unittest.main()
