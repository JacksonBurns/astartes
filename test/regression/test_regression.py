import os
import pickle as pkl
import unittest
from datetime import datetime

import numpy as np

from astartes import train_val_test_split
from astartes.samplers import (
    ALL_SAMPLERS,
    IMPLEMENTED_EXTRAPOLATION_SAMPLERS,
    IMPLEMENTED_INTERPOLATION_SAMPLERS,
)


class Test_regression(unittest.TestCase):
    """
    Test for regression relative to saved reference splits.
    """

    @classmethod
    def setUpClass(self):
        """Convenience attributes for later tests."""
        rng = np.random.default_rng(42)
        self.X = rng.random((100, 100))
        self.y = rng.random((100,))
        self.labels_datetime = np.array(
            [datetime.strptime(f"20{y:02}/01/01", "%Y/%m/%d") for y in range(100)]
        )
        cwd = os.getcwd()
        self.reference_splits_dir = os.path.join(
            cwd, "test", "regression", "reference_splits"
        )
        self.reference_splits = {
            name: os.path.join(self.reference_splits_dir, name + "_reference.pkl")
            for name in ALL_SAMPLERS
            if name not in ("scaffold",)
        }

    def test_timebased_regression(self):
        """Regression test TimeBased, which has labels to check as well."""
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            labels_train,
            labels_val,
            labels_test,
        ) = train_val_test_split(
            self.X,
            self.y,
            labels=self.labels_datetime,
            sampler="time_based",
            random_state=42,
        )
        all_output = [
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            labels_train,
            labels_val,
            labels_test,
        ]
        with open(self.reference_splits["time_based"], "rb") as f:
            reference_output = pkl.load(f)
        for i, j in zip(all_output, reference_output):
            np.testing.assert_array_equal(
                i, j, "Sampler time_based failed regression testing."
            )

    def test_interpolation_regression(self):
        """Regression testing of interpolative methods relative to static results."""
        for sampler_name in IMPLEMENTED_INTERPOLATION_SAMPLERS:
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
                self.X,
                self.y,
                sampler=sampler_name,
                random_state=42,
            )
            all_output = [X_train, X_val, X_test, y_train, y_val, y_test]
            with open(self.reference_splits[sampler_name], "rb") as f:
                reference_output = pkl.load(f)
            for i, j in zip(all_output, reference_output):
                np.testing.assert_array_equal(
                    i, j, "Sampler {:s} failed regression testing.".format(sampler_name)
                )

    def test_extrapolation_regression(self):
        """Regression testing of extrapolative methods relative to static results."""
        for sampler_name in IMPLEMENTED_EXTRAPOLATION_SAMPLERS:
            if sampler_name in ("scaffold", "time_based"):
                continue
            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                clusters_train,
                clusters_val,
                clusters_test,
            ) = train_val_test_split(
                self.X,
                self.y,
                sampler=sampler_name,
                random_state=42,
                hopts={"eps": 3.55},  # eps is used by DBSCAN
            )
            all_output = [
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                clusters_train,
                clusters_val,
                clusters_test,
            ]
            with open(self.reference_splits[sampler_name], "rb") as f:
                reference_output = pkl.load(f)
            for i, j in zip(all_output, reference_output):
                np.testing.assert_array_equal(
                    i, j, "Sampler {:s} failed regression testing.".format(sampler_name)
                )


if __name__ == "__main__":
    unittest.main()
