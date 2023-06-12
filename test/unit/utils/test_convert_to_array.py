import os
import sys
import unittest

import numpy as np
import pandas as pd

from astartes import train_test_split, train_val_test_split
from astartes.utils.exceptions import InvalidConfigurationError, UncastableInputError
from astartes.utils.warnings import ConversionWarning


class Test_convert_to_array(unittest.TestCase):
    """
    Test array type handling.
    """

    def test_panda_handla(self):
        """Splitting Dataframes and series should return them as such."""
        X = pd.DataFrame([[1], [2], [3]], index=[1, 2, 3], columns=["feature"])
        y = pd.Series([1, 2, 3], index=[1, 2, 3])
        train_X, test_X, train_y, test_y = train_test_split(
            X,
            y,
            train_size=0.67,
            test_size=0.33,
            sampler="kennard_stone",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                train_X.columns,
                X.columns,
            ),
            "astartes should preserve dataframe column names",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                train_X.index,
                [1, 3],
            ),
            "preserve indices of dataframes",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                test_X.columns,
                X.columns,
            ),
            "astartes should preserve dataframe column names",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                test_X.index,
                [2],
            ),
            "preserve indices of dataframes",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                test_y.index,
                [2],
            ),
            "preserve indices of series",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                train_y.index,
                [1, 3],
            ),
            "preserve indices of series",
        )

    def test_bad_type_cast(self):
        """Raise error when casting arrays that do not contain supported types."""
        with self.assertRaises(InvalidConfigurationError):
            train_val_test_split(
                ["cat", "dog"],
            )

    def test_convertable_input(self):
        """Raise warning when casting."""
        with self.assertWarns(ConversionWarning):
            train_val_test_split(
                [[1, 2], [3, 4], [5, 6], [7, 8]],
                [1, 2, 3, 4],
                train_size=0.50,
                test_size=0.25,
                val_size=0.25,
            )

    @unittest.skipIf(
        sys.version_info.minor == 7,
        "Versions of numpy compatible with Python 3.7 will convert ANYTHING "
        "into an array (thus the warning in convert_to_array, and skip test)",
    )
    def test_unconvertable_input(self):
        """Raise error when casting fails."""
        with self.assertRaises(UncastableInputError):
            # inhomogeneous lists w/ mixed types cannot be cast to arrays
            train_val_test_split(
                [[1], [1, 2]],
            )


if __name__ == "__main__":
    unittest.main()
