"""
This sampler partitions the data based on molecular weight. It first sorts the
molecules by molecular weight and then places the smallest molecules in the training set,
the next smallest in the validation set if applicable, and finally the largest molecules
in the testing set.
"""

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:
    # this is in place so that the import of this from parent directory will work
    # if it fails, it is caught in molecules instead and the error is more helpful
    NO_MOLECULES = True

from astartes.samplers import AbstractSampler


class MolecularWeight(AbstractSampler):
    def _before_sample(self):
        # ensure that X contains entries that are either a SMILES or InChI string or an RDKit Molecule
        if not all(isinstance(i, str) for i in self.X) and not all(
            isinstance(i, Chem.rdchem.Mol) for i in self.X
        ):
            msg = "MolecularWeight class requires input X to be an iterable of SMILES strings, InChI strings, or RDKit Molecules"
            raise TypeError(msg)

    def _sample(self):
        """
        Implements the molecular weight sampler to create an extrapolation split.
        """

        data = [
            (self.str_to_mol(x), idx) for x, idx in zip(self.X, np.arange(len(self.X)))
        ]
        sorted_list = sorted(data, reverse=False)

        self._samples_idxs = np.array([idx for time, idx in sorted_list], dtype=int)

    def str_to_mol(self, string):
        """
        Converts an InChI or SMILES string to an RDKit molecule
        and then calculates the average molecular weight.

        Params:
            string: The InChI or SMILES string.

        Returns:
            The average molecular weight of the molecule
        """
        RDKIT_SMILES_PARSER_PARAMS = Chem.SmilesParserParams()
        if string.startswith("InChI"):
            mol = Chem.MolFromInchi(string, removeHs=True)
        else:
            mol = Chem.MolFromSmiles(string)

        # calculate the average molecular weight of the molecule
        mol_wt = Descriptors.MolWt(mol)

        return mol_wt
