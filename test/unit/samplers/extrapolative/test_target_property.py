import unittest

import numpy as np

from astartes import train_test_split
from astartes.samplers import TargetProperty


class Test_TargetProperty(unittest.TestCase):
    """
    Test the various functionalities of TargetProperty.
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

    def test_target_property_sampling_ascending(self):
        """Use TargetProperty in the train_test_split and verify results."""
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
            sampler="target_property",
            hopts={"descending": False},
        )

        # test that the known arrays equal the result from above
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_train,
                self.X[:8],  # X was already sorted by ascending target value
            ),
            "Train X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_test,
                self.X[8:],  # X was already sorted by ascending target value
            ),
            "Test X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_train,
                self.y[:8],  # y was already sorted by ascending target value
            ),
            "Train y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_test,
                self.y[8:],  # y was already sorted by ascending target value
            ),
            "Test y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_train,
                self.labels[:8],  # labels was already sorted by ascending target value
            ),
            "Train labels incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_test,
                self.labels[8:],  # labels was already sorted by ascending target value
            ),
            "Test labels incorrect.",
        )

    def test_target_property_sampling_descending(self):
        """Use TargetProperty in the train_test_split and verify results."""
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
            sampler="target_property",
            hopts={"descending": True},
        )

        # test that the known arrays equal the result from above
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_train,
                np.flip(self.X)[:8],
            ),
            "Train X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_test,
                np.flip(self.X)[8:],
            ),
            "Test X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_train,
                np.flip(self.y)[:8],
            ),
            "Train y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                y_test,
                np.flip(self.y)[8:],
            ),
            "Test y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_train,
                np.flip(self.labels)[:8],
            ),
            "Train labels incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_test,
                np.flip(self.labels)[8:],
            ),
            "Test labels incorrect.",
        )

    def test_target_property(self):
        """Directly instantiate and test TargetProperty."""
        target_property_instance = TargetProperty(
            self.X,
            self.y,
            self.labels,
            {},
        )
        self.assertIsInstance(
            target_property_instance,
            TargetProperty,
            "Failed instantiation.",
        )
        self.assertFalse(
            len(target_property_instance.get_clusters()),
            "Clusters was set when it should not have been.",
        )
        self.assertTrue(
            len(target_property_instance._samples_idxs),
            "Sample indices not set.",
        )
