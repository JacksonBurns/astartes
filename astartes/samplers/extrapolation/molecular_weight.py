"""
This sampler partitions the data based on molecular weight. It first sorts the
molecules by molecular weight and then places the smallest molecules in the training set,
the next smallest in the validation set if applicable, and finally the largest molecules
in the testing set.
"""

import numpy as np

try:
    from astartes.utils.aimsim_featurizer import featurize_molecules
except ImportError:
    # this is in place so that the import of this from parent directory will work
    # if it fails, it is caught in molecules instead and the error is more helpful
    NO_MOLECULES = True

from .scaffold import Scaffold
from .target_property import TargetProperty


# inherit sample method from TargetProperty
class MolecularWeight(TargetProperty):
    def _before_sample(self):
        # check for invalid data types using the method in the Scaffold sampler
        Scaffold._validate_input(self.X)
        # calculate the average molecular weight of the molecule
        self.y_backup = self.y
        self.y = featurize_molecules((Scaffold.str_to_mol(i) for i in self.X), "mordred:MW", fprints_hopts={})

    def _after_sample(self):
        # restore the original y values
        self.y = self.y_backup
