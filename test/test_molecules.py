import os
import sys
import unittest
import warnings

import numpy as np
from rdkit import Chem

from astartes.molecules import (
    train_test_split_molecules,
    train_val_test_split_molecules,
)
from astartes.samplers import (
    IMPLEMENTED_EXTRAPOLATION_SAMPLERS,
    IMPLEMENTED_INTERPOLATION_SAMPLERS,
)
from astartes.utils.warnings import ImperfectSplittingWarning


class Test_molecules(unittest.TestCase):
    """
    Test the various functionalities of molecules.

    Note: daylight_fingerprint is not compatible -- inhomogenous arrays
    (variable length descriptor)
    """

    @classmethod
    def setUpClass(self):
        """Convenience attributes for later tests."""
        with open(os.path.join("test", "data", "qm9_smiles.txt"), "r") as file:
            lines = file.readlines()

        qm9_smiles_short = [i.replace("\n", "") for i in lines[:100]]
        qm9_smiles_full = [i.replace("\n", "") for i in lines]

        self.X = np.array(qm9_smiles_short)
        self.y = np.array(list(range(len(qm9_smiles_short))))

        molecule_array = []
        for smile in qm9_smiles_short:
            molecule_array.append(Chem.MolFromSmiles(smile))
        self.molecules = np.array(molecule_array)

        self.X_long = np.array(qm9_smiles_full)
        self.y_long = np.array(list(range(len(qm9_smiles_full))))

    def test_molecules(self):
        """Try train_test_split molecules with every interpolative sampler."""
        for sampler in IMPLEMENTED_INTERPOLATION_SAMPLERS:
            tts = train_test_split_molecules(
                self.X,
                self.y,
                train_size=0.2,
                sampler=sampler,
                fprints_hopts={"n_bits": 100},
            )

    def test_molecules_with_rdkit(self):
        """Try train_test_split molecules, every sampler, passing rdkit objects."""
        for sampler in IMPLEMENTED_INTERPOLATION_SAMPLERS:
            tts = train_test_split_molecules(
                self.molecules,
                self.y,
                train_size=0.2,
                sampler=sampler,
                fprints_hopts={"n_bits": 100},
            )

    def test_molecules_with_troublesome_smiles(self):
        """Helpful errors when rdkit graphs can't be featurized."""
        with self.assertRaises(RuntimeError):
            tts = train_test_split_molecules(
                np.array(["Nc1ncnc2n(cnc12)[C@@H]3O[C@H](CN=[N]=N)[C@@H](O)[C@H]3O"]),
                train_size=0.2,
                sampler="random",
                fprints_hopts={"n_bits": 100},
            )

    def test_validation_split_molecules(self):
        """Try train_val_test_split_molecule with every extrapolative sampler."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for sampler in IMPLEMENTED_EXTRAPOLATION_SAMPLERS:
                if sampler == "scaffold":
                    continue
                tts = train_val_test_split_molecules(
                    self.X,
                    self.y,
                    sampler=sampler,
                    hopts={"eps": 5},
                    fprints_hopts={"n_bits": 1000},
                )

    def test_fingerprints(self):
        """Test using different fingerprints with the molecular featurization."""
        for fprint in [
            "morgan_fingerprint",
            "topological_fingerprint",
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tts = train_test_split_molecules(
                    self.X,
                    self.y,
                    test_size=0.2,
                    sampler="random",
                    fingerprint=fprint,
                )

    def test_fprint_hopts(self):
        """Test specifying hyperparameters for the molecular featurization step."""
        tts = train_test_split_molecules(
            self.X,
            self.y,
            train_size=0.2,
            sampler="random",
            fingerprint="topological_fingerprint",
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
        """Test ability to pass through sampler hopts with molecules interface, expecting no warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            tts = train_test_split_molecules(
                self.X,
                self.y,
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

    def test_maximum_call(self):
        """Specify ALL the optional hyperparameters!"""
        tts = train_test_split_molecules(
            self.X,
            self.y,
            train_size=0.2,
            fingerprint="topological_fingerprint",
            fprints_hopts={
                "minPath": 2,
                "maxPath": 5,
                "fpSize": 200,
                "bitsPerHash": 4,
                "useHs": 1,
                "tgtDensity": 0.4,
                "minSize": 64,
            },
            sampler="random",
            random_state=42,
        )


if __name__ == "__main__":
    unittest.main()
