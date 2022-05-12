import os
import sys
import unittest

import numpy as np

from astartes import train_test_split


class Test_astartes(unittest.TestCase):
    """
    Test the various functionalities of astartes.
    """

    @classmethod
    def setUpClass(self):
        return

    def test_train_test_split(self):
        """
        """
        X_train, X_test, y_train, y_test = train_test_split(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            np.array([10, 11, 12]),
            test_size=0.2,
            train_size=0.8,
            splitter='random',
            hopts={
                'shuffle': True,
                'random_state': 42,
            }
        )
        for elt, ans in zip(X_train.flatten(), [4, 5, 6, 7, 8, 9]):
            self.assertEqual(elt, ans)


if __name__ == '__main__':
    unittest.main()
