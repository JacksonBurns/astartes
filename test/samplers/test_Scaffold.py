import unittest

import numpy as np

from astartes import train_test_split
from astartes.samplers import Scaffold


class Test_scaffold(unittest.TestCase):
    """
    Test the various functionalities of Scaffold.
    """

    @classmethod
    def setUpClass(self):
        """Convenience attributes for later tests."""
        self.X = np.array(
            [
                "c1ccccc1",
                "O=C1NCCO1",
                "O=C1CCCCCN1",
                "C1CCNCC1",
            ]
        )
        self.y = np.array([0, 1, 2, 3])
        self.labels = np.array(
            [
                "zero",
                "one",
                "two",
                "three",
            ]
        )

    def test_scaffold_sampling(self):
        """Use Scaffold in the train_test_split and verify results."""
        (
            X_train,
            X_test,
            y_train,
            y_test,
            labels_train,
            labels_test,
            clusters_train,
            clusters_test,
        ) = train_test_split(
            self.X,
            self.y,
            labels=self.labels,
            test_size=0.25,
            train_size=0.75,
            sampler="scaffold",
            hopts={},
        )

        # test that the known arrays equal the result from above
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_train,
                np.array(["O=C1NCCO1", "O=C1CCCCCN1", "C1CCNCC1"]),
            ),
            "Train X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_test,
                np.array(["c1ccccc1"]),
            ),
            "Test X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_train,
                np.array([1, 2, 3]),
            ),
            "Train y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_test,
                np.array([0]),
            ),
            "Test y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_train,
                np.array(["one", "two", "three"]),
            ),
            "Train labels incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_test,
                np.array(["zero"]),
            ),
            "Test labels incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                clusters_train,
                np.array([1, 2, 3]),
            ),
            "Train clusters incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                clusters_test,
                np.array([0]),
            ),
            "Test clusters incorrect.",
        )

    def test_scaffold(self):
        """Directly instantiate and test Scaffold."""
        scaffold_instance = Scaffold(
            self.X,
            self.y,
            self.labels,
            {},
        )
        self.assertIsInstance(
            scaffold_instance,
            Scaffold,
            "Failed instantiation.",
        )
        self.assertTrue(
            len(scaffold_instance.get_clusters()),
            "Clusters not set.",
        )
        self.assertTrue(
            len(scaffold_instance.get_sorted_cluster_counter()),
            "Sorted cluster Counter not found.",
        )
        self.assertTrue(
            len(scaffold_instance._samples_idxs),
            "Sample indices not set.",
        )

    def test_incorrect_input(self):
        """Calling with something other than SMILES should raise TypeError"""

    def test_no_scaffold_found_warning(self):
        """Molecules that cannot be scaffolded should raise a warning"""

    def test_mol_from_inchi(self):
        """Ability to load data from InChi inputs"""

    def test_explicit_hydrogens(self):
        """Include H in scaffold calculation"""


if __name__ == "__main__":
    unittest.main()
