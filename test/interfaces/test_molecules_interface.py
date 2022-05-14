import os
import sys
import unittest

import numpy as np

from astartes.interfaces import train_test_split_molecules
from astartes import IMPLEMENTED_SAMPLERS


class Test_molecules_interface(unittest.TestCase):
    """
    Test the various functionalities of molecules_interface.
    """

    @classmethod
    def setUpClass(self):

        with open(os.path.join("test", "data", "qm9_smiles.txt"), "r") as file:
            lines = file.readlines()

        qm9_smiles_short = [i.replace('\n', '') for i in lines[:500]]
        qm9_smiles_full = [i.replace('\n', '') for i in lines]

        self.X = qm9_smiles_short
        self.y = list(range(len(qm9_smiles_short)))

        self.X_long = qm9_smiles_full
        self.y_long = list(range(len(qm9_smiles_full)))

    def test_molecules_interface(self):
        """
        """
        for sampler in IMPLEMENTED_SAMPLERS:
            tts = train_test_split_molecules(
                self.X,
                self.y,
                0.2,
                splitter=sampler,
                fprints_hopts={'n_bits': 100},
            )

    def test_fingerprints(self):
        pass

    def test_fprint_hopts(self):
        pass

    def test_sampler_hopts(self):
        pass


if __name__ == '__main__':
    unittest.main()
