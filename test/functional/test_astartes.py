import os
import sys
import unittest
import warnings

import numpy as np

from astartes import train_test_split, train_val_test_split
from astartes.samplers import (
    IMPLEMENTED_EXTRAPOLATION_SAMPLERS,
    IMPLEMENTED_INTERPOLATION_SAMPLERS,
)
from astartes.utils.exceptions import (
    InvalidConfigurationError,
    SamplerNotImplementedError,
)
from astartes.utils.warnings import ImperfectSplittingWarning, NormalizationWarning


class Test_astartes(unittest.TestCase):
    """
    Test the various functionalities of astartes.
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
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        self.y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.labels = np.array(
            [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
            ]
        )

    def test_inconsistent_input_lengths(self):
        """Different length X, y, and labels should raise an exception at start."""
        with self.assertRaises(InvalidConfigurationError):
            train_val_test_split(
                np.array([1, 2]),
                np.array([1]),
                np.array([1, 2, 3]),
            )

    def test_train_val_test_split(self):
        """Split data into training, validation, and test sets."""
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                labels_train,
                labels_val,
                labels_test,
            ) = train_val_test_split(
                self.X,
                self.y,
                labels=self.labels,
                test_size=0.2,
                val_size=0.2,
                train_size=0.6,
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
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_train,
                np.array(
                    [
                        [1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                    ]
                ),
                "Train X incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_val,
                np.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
                "Validation X incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_test,
                np.array([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]]),
                "Test X incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_train,
                np.array([2, 6, 1, 8, 3, 10]),
                "Train y incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_val,
                np.array([5, 4]),
                "Validation y incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_test,
                np.array([7, 9]),
                "Test y incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_train,
                np.array(["two", "six", "one", "eight", "three", "ten"]),
                "Train labels incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_val,
                np.array(["five", "four"]),
                "Validation labels incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_test,
                np.array(["seven", "nine"]),
                "Test labels incorrect.",
            )
        )

    def test_train_val_test_split_extrpolation_shuffling(self):
        """Split data into training, validation, and test sets with shuffling."""
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                labels_train,
                labels_val,
                labels_test,
                clusters_train,
                clusters_val,
                clusters_test,
            ) = train_val_test_split(
                self.X,
                self.y,
                labels=self.labels,
                test_size=0.1,
                val_size=0.1,
                train_size=0.8,
                sampler="sphere_exclusion",
                random_state=867_5309,
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
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_train,
                np.array(
                    [
                        [1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0],
                        [1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 0],
                    ]
                ),
                "Train X incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_val,
                np.array([[0, 0, 0, 0, 0]]),
                "Validation X incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_test,
                np.array([[1, 1, 1, 1, 1]]),
                "Test X incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_train,
                np.array([2, 6, 3, 7, 4, 8, 5, 9]),
                "Train y incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_val,
                np.array([1]),
                "Validation y incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_test,
                np.array([10]),
                "Test y incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_train,
                np.array(
                    ["two", "six", "three", "seven", "four", "eight", "five", "nine"]
                ),
                "Train labels incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_val,
                np.array(["one"]),
                "Validation labels incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_test,
                np.array(["ten"]),
                "Test labels incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                clusters_train,
                np.array([4, 4, 0, 0, 1, 1, 5, 5]),
                "Train cluster assignments incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                clusters_val,
                np.array([2]),
                "Validation cluster assignments incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                clusters_test,
                np.array([3]),
                "Test cluster assignments incorrect.",
            )
        )

    def test_insufficient_dataset_train(self):
        """If the user requests a split that would result in rounding down the size of the
        test set to zero, a helpful exception should be raised."""
        with self.assertRaises(InvalidConfigurationError):
            train_val_test_split(
                self.X,
                train_size=None,  # this will result in an empty train set due to rounding
                val_size=0.4,
                test_size=0.6,
            )

    def test_insufficient_dataset_val(self):
        """If the user requests a split that would result in rounding down the size of the
        test set to zero, a helpful exception should be raised."""
        with self.assertRaises(InvalidConfigurationError):
            train_val_test_split(
                self.X,
                train_size=0.45,
                val_size=0.05,  # this will result in an empty val set due to rounding
                test_size=0.5,
            )

    def test_insufficient_dataset_test(self):
        """If the user requests a split that would result in rounding down the size of the
        test set to zero, a helpful exception should be raised."""
        with self.assertRaises(InvalidConfigurationError):
            train_val_test_split(
                self.X,
                train_size=0.49,
                val_size=0.5,
                test_size=0.01,  # empty test due to rounding
                sampler="kmeans",
            )

    def test_split_validation(self):
        """Tests of the input split validation."""
        with self.subTest("no splits specified"):
            with self.assertRaises(RuntimeError):
                train_val_test_split(
                    self.X,
                    train_size=None,
                    val_size=None,
                    test_size=None,
                )
        with self.subTest("all splits specified"):
            train_val_test_split(
                self.X,
                train_size=0.2,
                val_size=0.2,
                test_size=0.6,
            )
        with self.subTest("all specified imperfectly"):
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                train_val_test_split(
                    self.X,
                    train_size=20,
                    val_size=0.2,
                    test_size=60,
                )
                self.assertTrue(
                    len(w) == 2,
                    "\nTwo warnings should have been raised when requesting a mathematically impossible split."
                    "\nReceived {:d} warnings instead: \n -> {:s}".format(
                        len(w),
                        "\n -> ".join(
                            [
                                str(i.category.__name__) + ": " + str(i.message)
                                for i in w
                            ]
                        ),
                    ),
                )

        with self.subTest("invalid val_size"):
            with self.assertRaises(InvalidConfigurationError):
                train_val_test_split(
                    self.X,
                    train_size=20,
                    val_size=20,
                    test_size=60,
                )
        with self.subTest("no val_size w/ test and train"):
            with self.assertWarns(NormalizationWarning):
                train_val_test_split(
                    self.X,
                    train_size=7,
                    val_size=None,
                    test_size=3,
                )
        with self.subTest("invalid val_size"):
            with self.assertRaises(RuntimeError):
                train_val_test_split(
                    self.X,
                    train_size=0.1,
                    val_size=42,
                    test_size=None,
                )
        with self.subTest("val_size w/ valid test_size"):
            with self.assertWarns(ImperfectSplittingWarning):
                train_val_test_split(
                    self.X,
                    train_size=None,
                    val_size=0.4,
                    test_size=0.2,
                )
        with self.subTest("val_size w/ invalid test_size"):
            with self.assertRaises(RuntimeError):
                train_val_test_split(
                    self.X,
                    train_size=None,
                    val_size=0.4,
                    test_size=2,
                )
        with self.subTest("val_size w/ valid train_size"):
            train_val_test_split(
                self.X,
                train_size=0.2,
                val_size=0.4,
                test_size=None,
            )
        with self.subTest("val_size w/ invalid train_size"):
            with self.assertRaises(RuntimeError):
                train_val_test_split(
                    self.X,
                    train_size=12,
                    val_size=0.4,
                    test_size=None,
                )
        with self.subTest("no val_size w/ valid test_size"):
            train_val_test_split(
                self.X,
                train_size=None,
                val_size=None,
                test_size=0.5,
            )
        with self.subTest("no val_size w/ invalid test_size"):
            with self.assertRaises(RuntimeError):
                train_val_test_split(
                    self.X,
                    train_size=None,
                    val_size=None,
                    test_size=-2,
                )
        with self.subTest("no val_size w/ valid train_size"):
            train_val_test_split(
                self.X,
                train_size=0.6,
                val_size=None,
                test_size=None,
            )
        with self.subTest("no val_size w/ invalid train_size"):
            with self.assertRaises(RuntimeError):
                train_val_test_split(
                    self.X,
                    train_size=42,
                    val_size=None,
                    test_size=None,
                )

    def test_close_mispelling_sampler(self):
        """Astartes should be helpful in the event of a typo."""
        with self.assertRaises(SamplerNotImplementedError) as e:
            train_test_split([], sampler="radnom")
            self.assertEqual(
                e.exception,
                "Sampler radnom has not been implemented. Did you mean 'random'?",
            )

    def test_not_implemented_sampler(self):
        """Astartes should suggest checking the docstring."""
        with self.assertRaises(SamplerNotImplementedError) as e:
            train_test_split([], sampler="MIT is overrated")
            self.assertEqual(
                e.exception,
                "Sampler radnom has not been implemented. Try help(train_test_split).",
            )

    def test_train_test_split(self):
        """Funational test of train_test_split with imperfect splitting."""
        with self.assertWarns(ImperfectSplittingWarning):
            (
                X_train,
                X_test,
                y_train,
                y_test,
                labels_train,
                labels_test,
            ) = train_test_split(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                np.array([10, 11, 12]),
                labels=np.array(["apple", "banana", "apple"]),
                test_size=0.5,
                train_size=0.5,
                sampler="random",
                random_state=42,
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    X_train,
                    np.array([[4, 5, 6]]),
                    "X_train incorrect.",
                ),
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    X_test,
                    np.array([[7, 8, 9], [1, 2, 3]]),
                    "X_test incorrect.",
                ),
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    y_train,
                    np.array([11]),
                    "y_train incorrect.",
                ),
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    y_test,
                    np.array([12, 10]),
                    "y_test incorrect.",
                ),
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    labels_train,
                    np.array(["banana"]),
                    "labels_train incorrect.",
                ),
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    labels_test,
                    np.array(["apple", "apple"]),
                    "labels_test incorrect.",
                ),
            )

    def test_return_indices(self):
        """Test the ability to return the indices and the values."""
        with self.assertWarns(ImperfectSplittingWarning):
            (
                train_X,
                test_X,
                train_y,
                test_y,
                train_labels,
                test_labels,
                train_indices,
                test_indices,
            ) = train_test_split(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                np.array([10, 11, 12]),
                labels=np.array(["apple", "banana", "apple"]),
                test_size=0.5,
                train_size=0.5,
                sampler="random",
                random_state=42,
                return_indices=True,
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    test_indices,
                    np.array([2, 0]),
                    "Test indices incorrect.",
                ),
            )

    def test_return_indices_with_validation(self):
        """Test the ability to return indices in train_val_test_split"""
        with self.assertWarns(ImperfectSplittingWarning):
            (
                train_X,
                val_X,
                test_X,
                train_y,
                val_y,
                test_y,
                train_labels,
                val_labels,
                test_labels,
                train_indices,
                val_indices,
                test_indices,
            ) = train_val_test_split(
                np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                labels=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                test_size=0.33,
                val_size=0.33,
                train_size=0.34,
                sampler="random",
                random_state=42,
                return_indices=True,
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    val_indices,
                    np.array([8, 2]),
                    "Validation indices incorrect.",
                ),
            )


if __name__ == "__main__":
    unittest.main()
