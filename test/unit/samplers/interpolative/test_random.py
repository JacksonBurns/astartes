import unittest
import warnings

import numpy as np

from astartes import train_test_split
from astartes.samplers.interpolation import Random
from astartes.utils.warnings import ImperfectSplittingWarning


class Test_random(unittest.TestCase):
    """
    Test the various functionalities of Random.
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
        )
        self.y = np.array([1, 2, 3])
        self.labels = np.array(
            [
                "one",
                "two",
                "three",
            ]
        )

    def test_random(self):
        """Directly instantiate and test random."""
        ks_instance = Random(
            self.X,
            self.y,
            self.labels,
            {"random_state": 42},
        )
        self.assertIsInstance(
            ks_instance,
            Random,
            "Failed instantiation.",
        )
        self.assertTrue(
            len(ks_instance._samples_idxs),
            "Sample indices not set.",
        )

    def test_random_sample(self):
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
                test_size=0.5,
                train_size=0.5,
                sampler="random",
                random_state=42,
            )
        # test that the known arrays equal the result from above
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_train,
                np.array([[0, 1, 0]]),
                "Train X incorrect.",
            ),
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_test,
                np.array([[1, 1, 1], [0, 0, 0]]),
                "Test X incorrect.",
            ),
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_train,
                np.array([2]),
                "Train y incorrect.",
            ),
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_test,
                np.array([3, 1]),
                "Test y incorrect.",
            ),
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_train,
                np.array(["two"]),
                "Train labels incorrect.",
            ),
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_test,
                np.array(["three", "one"]),
                "Test labels incorrect.",
            ),
        )

    def test_random_sample_no_warning(self):
        """Use random with a mathematically possible split requested"""
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
                sampler="random",
                random_state=42,
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
