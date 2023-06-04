import os
import sys
import unittest

import numpy as np

from astartes import train_val_test_split
from astartes.utils.exceptions import UncastableInputError
from astartes.utils.warnings import ConversionWarning


class Test_convert_to_array(unittest.TestCase):
    """
    Test convert to numpy array for failures.
    """

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

    def test_unconvertable_input(self):
        """Raise error when casting fails."""
        with self.assertRaises(UncastableInputError):
            train_val_test_split(
                [[1], [1, 2]],  # inhomogeneous lists cannot be cast to arrays
            )


if __name__ == "__main__":
    unittest.main()
