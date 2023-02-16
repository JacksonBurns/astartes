import os
import sys
import unittest

import numpy as np

from astartes import train_test_split
from astartes.samplers import ALL_SAMPLERS


class Test_exceptions(unittest.TestCase):
    """
    Test that exceptions are raised when appropriate.
    """

    @classmethod
    def setUpClass(self):
        return

    def test_train_test_split(self):
        """ """
        pass


if __name__ == "__main__":
    unittest.main()
