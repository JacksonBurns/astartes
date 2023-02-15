import unittest
import warnings

import numpy as np

from astartes import train_test_split
from astartes.samplers.interpolation import KennardStone
from astartes.utils.warnings import ImperfectSplittingWarning


class Test_kennard_stone(unittest.TestCase):
    """
    Test the various functionalities of kennard_stone.
    """

    @classmethod
    def setUpClass(self):
        """Save re-used arrays as class attributes."""
        self.X = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
            ]
        )  # KS will sample two furthest for train
        self.y = np.array([1, 2, 3])
        self.labels = np.array(
            [
                "one",
                "two",
                "three",
            ]
        )

    def test_kennard_stone(self):
        """Directly instantiate and test KennardStone."""
        ks_instance = KennardStone(
            self.X,
            self.y,
            self.labels,
            {},  # KS does not have hyperparameters
        )
        self.assertIsInstance(
            ks_instance,
            KennardStone,
            "Failed instantiation.",
        )
        self.assertFalse(
            len(ks_instance.get_clusters()),
            "Clusters was set when it should not have been.",
        )
        self.assertFalse(
            len(ks_instance.get_sorted_cluster_counter()),
            "Sorted cluster Counter found when it should not be.",
        )
        self.assertTrue(
            len(ks_instance._samples_idxs),
            "Sample indices not set.",
        )

    def test_kennard_stone_sample(self):
        """Use kennard stone in tts and verify results"""
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
                test_size=0.25,
                train_size=0.75,
                sampler="kennard_stone",
            )
        # test that the known arrays equal the result from above
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_train,
                np.array([[0, 0, 0], [1, 1, 1]]),
            ),
            "Train X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_test,
                np.array([[0, 1, 0]]),
            ),
            "Test X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_train,
                np.array([1, 3]),
            ),
            "Train y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_test,
                np.array([2]),
            ),
            "Test y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_train,
                np.array(["one", "three"]),
            ),
            "Train labels incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_test,
                np.array(["two"]),
            ),
            "Test labels incorrect.",
        )

    def test_kennard_stone_sample_no_warning(self):
        """Use kennard stone with a mathematically possible split requested"""
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
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
                test_size=0.33,
                train_size=0.67,
                sampler="kennard_stone",
            )
            self.assertFalse(
                len(w),
                "\nNo warnings should have been raised when requesting a mathematically possible split."
                "\nReceived {:d} warnings instead: \n -> {:s}".format(
                    len(w),
                    "\n -> ".join(
                        [str(i.category.__name__) + ": " + str(i.message) for i in w]
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
