import os
import sys
import unittest

import numpy as np

from astartes.samplers import (
    ALL_SAMPLERS,
    DETERMINISTIC_EXTRAPOLATION_SAMPLERS,
    AbstractSampler,
)
from astartes.utils.sampler_factory import SamplerFactory


class Test_sampler_factory(unittest.TestCase):
    """
    Test SamplerFactory functions on all samplers.
    """

    @classmethod
    def setUpClass(self):
        """Save re-used arrays as class attributes."""
        self.X = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
            ]
        )
        self.y = np.array([1, 2, 3])

    def test_train_test_split(self):
        """Call sampler factory on all inputs."""
        for sampler_name in ALL_SAMPLERS:
            if sampler_name in ("scaffold", *DETERMINISTIC_EXTRAPOLATION_SAMPLERS):
                continue
            test_factory = SamplerFactory(sampler_name)
            test_instance = test_factory.get_sampler(self.X, self.y, None, {})
            self.assertIsInstance(
                test_instance,
                AbstractSampler,
                "Sampler {:s} failed to instantiate in SamplerFactory.".format(
                    sampler_name,
                ),
            )


if __name__ == "__main__":
    unittest.main()
