import os
import sys
import unittest
import warnings

import numpy as np

from astartes.molecules import train_test_split_molecules
from astartes.samplers import (
    IMPLEMENTED_SUPERVISED_SAMPLERS,
    IMPLEMENTED_UNSUPERVISED_SAMPLERS,
)


class Test_molecules_interface(unittest.TestCase):
    """
    Test the various functionalities of molecules_interface.
    """

    @classmethod
    def setUpClass(self):

        with open(os.path.join("test", "data", "qm9_smiles.txt"), "r") as file:
            lines = file.readlines()

        qm9_smiles_short = [i.replace("\n", "") for i in lines[:100]]
        qm9_smiles_full = [i.replace("\n", "") for i in lines]

        self.X = qm9_smiles_short
        self.y = list(range(len(qm9_smiles_short)))

        self.X_long = qm9_smiles_full
        self.y_long = list(range(len(qm9_smiles_full)))

    def test_molecules_interface(self):
        """ """
        for sampler in IMPLEMENTED_UNSUPERVISED_SAMPLERS:
            tts = train_test_split_molecules(
                self.X,
                self.y,
                0.2,
                splitter=sampler,
                fprints_hopts={"n_bits": 100},
            )

    def test_fingerprints(self):
        for fprint in [
            "morgan_fingerprint",
            "topological_fingerprint",
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tts = train_test_split_molecules(
                    self.X,
                    self.y,
                    0.2,
                    splitter="random",
                    fingerprint=fprint,
                )

    def test_fprint_hopts(self):
        tts = train_test_split_molecules(
            self.X,
            self.y,
            0.2,
            splitter="random",
            fingerprint="daylight_fingerprint",
            fprints_hopts={
                "minPath": 2,
                "maxPath": 5,
                "fpSize": 200,
                "bitsPerHash": 4,
                "useHs": 1,
                "tgtDensity": 0.4,
                "minSize": 64,
            },
        )

    def test_sampler_hopts(self):
        tts = train_test_split_molecules(
            self.X,
            self.y,
            0.2,
            splitter="random",
            hopts={
                "random_state": 42,
                "shuffle": True,
            },
        )

    def test_maximum_call(self):
        tts = train_test_split_molecules(
            self.X,
            self.y,
            0.2,
            fingerprint="daylight_fingerprint",
            fprints_hopts={
                "minPath": 2,
                "maxPath": 5,
                "fpSize": 200,
                "bitsPerHash": 4,
                "useHs": 1,
                "tgtDensity": 0.4,
                "minSize": 64,
            },
            splitter="random",
            hopts={
                "random_state": 42,
                "shuffle": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
