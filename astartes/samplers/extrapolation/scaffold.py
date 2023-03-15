"""
This sampler partitions the data based on the Bemis-Murcko scaffold function as implemented in RDKit.
Bemis, G. W.; Murcko, M. A. The Properties of Known Drugs. 1. Molecular Frameworks. J. Med. Chem. 1996, 39, 2887−2893.
Landrum, G. et al. RDKit: Open-Source Cheminformatics; 2006; https://www.rdkit.org.

The goal is to cluster molecules that share the same scaffold.
Later, these clusters will be assigned to training, validation, and testing split
to create data splits that will measure extrapolation by testing on scaffolds
that are not in the training set.

"""
import warnings
from collections import defaultdict

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    # this is in place so that the import of this from parent directory will work
    # if it fails, it is caught in molecules instead and the error is more helpful
    NO_MOLECULES = True

from astartes.samplers import AbstractSampler
from astartes.utils.warnings import NoMatchingScaffold


class Scaffold(AbstractSampler):
    def __init__(self, *args):
        if not isinstance(args[0][0], str) and not isinstance(
            args[0][0], Chem.rdchem.Mol
        ):
            msg = "Scaffold class requires input X to be an iterable of SMILES strings"
            raise TypeError(msg)

        super().__init__(*args)

    def _sample(self):
        """Implements the Scaffold sampler to identify clusters via a molecule's Bemis-Murcko scaffold."""
        scaffold_to_indices = self.scaffold_to_smiles(self.X)

        cluster_indices = np.empty(len(self.X), dtype=object)
        # give each cluster an arbitrary ID
        for cluster_id, (scaffold, indices) in enumerate(scaffold_to_indices.items()):
            if scaffold == "":
                warnings.warn(
                    f"No matching scaffold was found for the {len(indices)} "
                    f"molecules corresponding to indices {indices}",
                    NoMatchingScaffold,
                )
            for idx in indices:
                cluster_indices[idx] = scaffold

        self._samples_clusters = cluster_indices

    def scaffold_to_smiles(self, mols):
        """
        Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

        Params:
            mols: A list of smiles strings or RDKit molecules.

        Returns:
            A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
        """
        scaffolds = defaultdict(set)
        for i, mol in enumerate(mols):
            scaffold = self.generate_bemis_murcko_scaffold(
                mol, self.get_config("include_chirality", False)
            )
            scaffolds[scaffold].add(i)

        return scaffolds

    def str_to_mol(self, string, explicit_hydrogens=False):
        """
        Converts an InChI or SMILES string to an RDKit molecule.

        Params:
            string: The InChI or SMILES string.
            explicit_hydrogens: Whether to treat hydrogens explicitly.

        Returns:
            An RDKit molecule.
        """
        RDKIT_SMILES_PARSER_PARAMS = Chem.SmilesParserParams()
        if string.startswith("InChI"):
            mol = Chem.MolFromInchi(string, removeHs=not explicit_hydrogens)
        else:
            # Set params here so we don't remove hydrogens with atom mapping
            RDKIT_SMILES_PARSER_PARAMS.removeHs = not explicit_hydrogens
            mol = Chem.MolFromSmiles(string, RDKIT_SMILES_PARSER_PARAMS)

        if explicit_hydrogens:
            return Chem.AddHs(mol)
        else:
            return Chem.RemoveHs(mol)

    def generate_bemis_murcko_scaffold(self, mol, include_chirality=False):
        """
        Compute the Bemis-Murcko scaffold for an RDKit molecule.

        Params:
            mol: A smiles string or an RDKit molecule.
            include_chirality: Whether to include chirality.

        Returns:
            Bemis-Murcko scaffold
        """
        mol = (
            self.str_to_mol(mol, self.get_config("explicit_hydrogens", False))
            if isinstance(mol, str)
            else mol
        )
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )

        return scaffold
