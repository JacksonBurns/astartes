import unittest

import numpy as np
from sklearn.svm import LinearSVR

from astartes.samplers.interpolation import Random
from astartes.utils.exceptions import InvalidModelTypeError
from astartes.utils import generate_regression_results_dict


class Test_utils(unittest.TestCase):
    """
    Test functions within utils.py.
    """

    @classmethod
    def setUpClass(self):
        """Save re-used arrays as class attributes."""
        # X and y come from sklearn's make_regression function
        self.X = np.array(
            [
                [-0.86, -0.98],
                [-0.42, -0.87],
                [1.33, 0.20],
                [-0.25, 2.43],
                [-0.59, -0.91],
                [-0.33, 0.19],
                [-0.10, -0.01],
                [1.86, 1.15],
                [0.64, -1.51],
                [-0.36, 0.06],
                [0.6, -0.36],
                [1.56, -0.09],
                [-0.70, -1.66],
                [-0.33, 0.44],
                [1.58, 0.11],
                [0.25, -0.05],
                [-0.63, 0.79],
                [-0.11, 0.00],
                [-0.20, -1.19],
                [0.71, 1.00],
            ]
        )
        self.y = np.array(
            [
                -10.27,
                -6.19,
                12.13,
                4.90,
                -7.77,
                -2.31,
                -0.89,
                19.42,
                1.18,
                -2.97,
                4.18,
                13.26,
                -10.90,
                -1.58,
                14.01,
                2.00,
                -3.16,
                -0.91,
                -5.25,
                9.07,
            ]
        )

    def test_generate_regression_results_dict(self):
        """Generate results dictionary for simple regression task."""

        # test that error is raised if not using sklearn model
        with self.assertRaises(InvalidModelTypeError) as e:
            generate_regression_results_dict(
                Random,
                self.X,
                self.y,
                train_size=0.6,
                val_size=0.2,
                test_size=0.2,
            )

        # use default hyperparameters
        sklearn_model = LinearSVR()

        # test function call and also that a table can be printed without error
        results_dict = generate_regression_results_dict(
            sklearn_model,
            self.X,
            self.y,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            print_results=True,
        )

        # test that only results for the default random sampler are included
        self.assertEqual(
            len(results_dict),
            1,
            msg=f"results_dict contained {results_dict.keys()}. Expected just random sampler.",
        )
        # test that results for mae, rmse, and r2 are included
        self.assertTrue(
            "mae" in results_dict["random"].keys(),
            msg=f"results_dict did not contain MAE results.",
        )
        self.assertTrue(
            "rmse" in results_dict["random"].keys(),
            msg=f"results_dict did not contain RMSE results.",
        )
        self.assertTrue(
            "R2" in results_dict["random"].keys(),
            msg=f"results_dict did not contain R2 results.",
        )


if __name__ == "__main__":
    unittest.main()
