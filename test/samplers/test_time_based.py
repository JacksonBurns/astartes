import unittest
from datetime import datetime

import numpy as np

from astartes import train_test_split
from astartes.samplers import TimeBased
from astartes.utils.warnings import ImperfectSplittingWarning


class Test_time_based(unittest.TestCase):
    """
    Test the various functionalities of TimeBased.
    """

    @classmethod
    def setUpClass(self):
        """Convenience attributes for later tests."""
        self.X = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
            ]
        )
        self.y = np.array([1, 2, 3, 4, 5])

        self.dates = [f'20{y:02}/01/01' for y in range(5)]
        self.labels_datetime = np.array([datetime.strptime(date, '%Y/%m/%d') for date in self.dates])
        self.labels_date = np.array([datetime.strptime(date, '%Y/%m/%d').date() for date in self.dates])

    def test_time_based_sampling(self):
        """Use time_based in the train_test_split and verify results."""
        with self.assertWarns(ImperfectSplittingWarning):
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
                labels=self.labels_datetime,
                test_size=0.25,
                train_size=0.75,
                sampler="time_based",
            )
        # test that the known arrays equal the result from above
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_train,
                np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0]]),
            ),
            "Train X incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                X_test,
                np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]),
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
                np.array([4, 5]),
            ),
            "Test y incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_train,
                np.array([datetime.strptime(date, '%Y/%m/%d') for date in self.dates[:3]])
            ),
            "Train labels incorrect.",
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                labels_test,
                np.array([datetime.strptime(date, '%Y/%m/%d') for date in self.dates[-2:]])
            ),
            "Test labels incorrect.",
        )

    def test_time_based_date(self):
        """Directly instantiate and test TimeBased."""
        time_based_instance = TimeBased(
            self.X,
            self.y,
            self.labels_date,
            {},  # TimeBased does not have hyperparameters
        )
        self.assertIsInstance(
            time_based_instance,
            TimeBased,
            "Failed instantiation.",
        )
        self.assertFalse(
            len(time_based_instance.get_clusters()),
            "Clusters was set when it should not have been.",
        )
        self.assertFalse(
            len(time_based_instance.get_sorted_cluster_counter()),
            "Sorted cluster Counter found when it should not be.",
        )
        self.assertTrue(
            len(time_based_instance._samples_idxs),
            "Sample indices not set.",
        )

    def test_time_based_datetime(self):
        """Directly instantiate and test TimeBased."""
        time_based_instance = TimeBased(
            self.X,
            self.y,
            self.labels_datetime,
            {},  # TimeBased does not have hyperparameters
        )
        self.assertIsInstance(
            time_based_instance,
            TimeBased,
            "Failed instantiation.",
        )
        self.assertFalse(
            len(time_based_instance.get_clusters()),
            "Clusters was set when it should not have been.",
        )
        self.assertFalse(
            len(time_based_instance.get_sorted_cluster_counter()),
            "Sorted cluster Counter found when it should not be.",
        )
        self.assertTrue(
            len(time_based_instance._samples_idxs),
            "Sample indices not set.",
        )

    def test_mising_labels(self):
        """Not specifying labels should raise ValueError"""
        with self.assertRaises(ValueError):
            train_test_split(
                X=self.X,
                sampler="time_based",
            )

    def test_incorrect_input(self):
        """Specifying labels as neither date nor datetime object should raise TypeError"""
        with self.assertRaises(TypeError):
            train_test_split(
                X=self.X,
                labels=np.arange(5),
                sampler="time_based",
            )


if __name__ == "__main__":
    unittest.main()
