import os
import sys
import unittest

import numpy as np

from astartes.interfaces import train_test_split_molecules
from astartes import IMPLEMENTED_SAMPLERS
from ..data import qm9_smiles_short


class Test_molecules_interface(unittest.TestCase):
    """
    Test the various functionalities of molecules_interface.
    """

    @classmethod
    def setUpClass(self):
        self.X = qm9_smiles_short
        self.y = list(range(len(qm9_smiles_short)))

    def test_molecules_interface(self):
        """
        """
        for sampler in IMPLEMENTED_SAMPLERS:
            tts = train_test_split_molecules(
                self.X,
                self.y,
                0.2,
                splitter=sampler,
            )


if __name__ == '__main__':
    unittest.main()
