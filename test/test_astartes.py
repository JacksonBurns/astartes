import os
import sys
import unittest
import warnings

import numpy as np

from astartes import train_test_split
from astartes import train_val_test_split
from astartes.samplers import (
    IMPLEMENTED_EXTRAPOLATION_SAMPLERS,
    IMPLEMENTED_INTERPOLATION_SAMPLERS,
)
from astartes.utils.exceptions import NotImplementedError
from astartes.utils.warnings import ImperfectSplittingWarning


class Test_astartes(unittest.TestCase):
    """
    Test the various functionalities of astartes.
    """

    @classmethod
    def setUpClass(self):
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
                val_size = 0.2,
                train_size=0.6,
                sampler="random",
                hopts={
                    "random_state": 42,
                },
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
                np.array([[0, 1, 0], [1, 1, 1]]),
                "Train X incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_val,
                np.array([[0, 0, 0]]),
                "Validation X incorrect.",
            )            
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_test,
                np.array([[0, 0, 0]]),
                "Test X incorrect.",
            )            
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_train,
                np.array([2, 3]),
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
                np.array([1]),
                "Test y incorrect.",
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_train,
                np.array(["two", "three"]),
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
                np.array(["one"]),
                "Test labels incorrect.",
            )
        )

    def test_insufficient_dataset(self):
        """If the user requests a split that would result in rounding down the size of the
        test set to zero, a helpful exception should be raised."""

    def test_rounding_warning(self):
        """astartes should warn when normalizing floats that do not add to 1 or ints that do
        not add to len(X) when providing test_size and train_size in tts."""

    def test_close_mispelling_sampler(self):
        """Astartes should be helpful in the event of a typo."""
        with self.assertRaises(NotImplementedError) as e:
            train_test_split([], sampler="radnom")
            self.assertEqual(
                e.exception,
                "Sampler radnom has not been implemented. Did you mean 'random'?",
            )

    def test_not_implemented_sampler(self):
        """Astartes should suggest checking the docstring."""
        with self.assertRaises(NotImplementedError) as e:
            train_test_split([], sampler="MIT is overrated")
            self.assertEqual(
                e.exception,
                "Sampler radnom has not been implemented. Try help(train_test_split).",
            )

    def test_train_test_split(self):
        """ """
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
                test_size=0.3,
                train_size=0.7,
                sampler="random",
                hopts={
                    "random_state": 42,
                },
            )
            for elt, ans in zip(X_train.flatten(), [4, 5, 6, 7, 8, 9]):
                self.assertEqual(elt, ans)

    def test_return_indices(self):
        """ """
        with self.assertWarns(ImperfectSplittingWarning):
            (indices_train, indices_test,) = train_test_split(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                np.array([10, 11, 12]),
                labels=np.array(["apple", "banana", "apple"]),
                test_size=0.3,
                train_size=0.7,
                sampler="random",
                hopts={
                    "random_state": 42,
                },
                return_indices=True,
            )
            for elt, ans in zip(indices_train.flatten(), [1, 2]):
                self.assertEqual(elt, ans)


if __name__ == "__main__":
    unittest.main()
