import os
import sys
import unittest

import numpy as np

from astartes import train_test_split
from astartes.utils.warnings import (
    ImperfectSplittingWarning,
)
from astartes.samplers import (
    IMPLEMENTED_INTERPOLATION_SAMPLERS,
    IMPLEMENTED_EXTRAPOLATION_SAMPLERS,
)


class Test_astartes(unittest.TestCase):
    """
    Test the various functionalities of astartes.
    """

    @classmethod
    def setUpClass(self):
        return

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
            for elt, ans in zip(X_train.flatten(), [4, 5, 6, 1, 2, 3]):
                self.assertEqual(elt, ans)

    def test_return_indices(self):
        """ """
        pass


if __name__ == "__main__":
    unittest.main()
