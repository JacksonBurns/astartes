import unittest

import numpy as np

from astartes import train_test_split

aimsim = None
REASON = None
try:
    import aimsim

    from astartes.samplers import MolecularWeight
except ModuleNotFoundError:
    REASON = "molecules subpackage not installed"


@unittest.skipIf(aimsim is None, reason=REASON)
class Test_MolecularWeight(unittest.TestCase):
    """
    Test the various functionalities of MolecularWeight.
    """

    @classmethod
    def setUpClass(self):
        """Convenience attributes for later tests."""
        self.X = np.array(
            [
                "C",
                "CC",
                "CCC",
                "CCCC",
                "CCCCC",
                "CCCCCC",
                "CCCCCCC",
                "CCCCCCCC",
                "CCCCCCCCC",
                "CCCCCCCCCC",
            ]
        )
        self.X_inchi = np.array(
            [
                "InChI=1S/CH4/h1H4",
                "InChI=1S/C2H6/c1-2/h1-2H3",
                "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3",
                "InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3",
                "InChI=1S/C5H12/c1-3-5-4-2/h3-5H2,1-2H3",
                "InChI=1S/C6H14/c1-3-5-6-4-2/h3-6H2,1-2H3",
                "InChI=1S/C7H16/c1-3-5-7-6-4-2/h3-7H2,1-2H3",
                "InChI=1S/C8H18/c1-3-5-7-8-6-4-2/h3-8H2,1-2H3",
                "InChI=1S/C9H20/c1-3-5-7-9-8-6-4-2/h3-9H2,1-2H3",
                "InChI=1S/C10H22/c1-3-5-7-9-10-8-6-4-2/h3-10H2,1-2H3",
            ]
        )
        self.y = np.arange(len(self.X))
        self.labels = np.array(
            [
                "methane",
                "ethane",
                "propane",
                "butane",
                "pentane",
                "hexane",
                "heptane",
                "octane",
                "nonane",
                "decane",
            ]
        )

    def test_molecular_weight_sampling(self):
        """Use MolecularWeight in the train_test_split and verify results."""
        (
            X_train,
            X_test,
            y_train,
            y_test,
            labels_train,
            labels_test,
        ) = train_test_split(
            self.X,
            self.y,
            labels=self.labels,
            test_size=0.2,
            train_size=0.8,
            sampler="molecular_weight",
            hopts={},
        )

        # test that the known arrays equal the result from above
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_train,
                self.X[:8],  # X was already sorted by ascending molecular weight
            ),
            "Train X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_test,
                self.X[8:],  # X was already sorted by ascending molecular weight
            ),
            "Test X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_train,
                self.y[:8],  # y was already sorted by ascending molecular weight
            ),
            "Train y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_test,
                self.y[8:],  # y was already sorted by ascending molecular weight
            ),
            "Test y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_train,
                self.labels[:8],  # labels was already sorted by ascending molecular weight
            ),
            "Train labels incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_test,
                self.labels[8:],  # labels was already sorted by ascending molecular weight
            ),
            "Test labels incorrect.",
        )

    def test_molecular_weight(self):
        """Directly instantiate and test MolecularWeight."""
        molecular_weight_instance = MolecularWeight(
            self.X,
            self.y,
            self.labels,
            {},
        )
        self.assertIsInstance(
            molecular_weight_instance,
            MolecularWeight,
            "Failed instantiation.",
        )
        self.assertFalse(
            len(molecular_weight_instance.get_clusters()),
            "Clusters was set when it should not have been.",
        )
        self.assertTrue(
            len(molecular_weight_instance._samples_idxs),
            "Sample indices not set.",
        )

    def test_incorrect_input(self):
        """Calling with something other than SMILES, InChI, or RDKit Molecule should raise TypeError"""
        with self.assertRaises(TypeError):
            train_test_split(
                np.array([[1], [2]]),
                sampler="molecular_weight",
            )

    def test_mol_from_inchi(self):
        """Ability to load data from InChi inputs"""
        MolecularWeight(
            self.X_inchi,
            None,
            None,
            {},
        )


if __name__ == "__main__":
    unittest.main()
