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
    def _validate_input(X):
        # ensure that X contains entries that are either a SMILES string or an RDKit Molecule
        if not all(isinstance(i, str) for i in X) and not all(isinstance(i, Chem.rdchem.Mol) for i in X):
            msg = "Scaffold class requires input X to be an iterable of SMILES strings, InChI strings, or RDKit Molecules"
            raise TypeError(msg)

    def _before_sample(self):
        Scaffold._validate_input(self.X)

    def _sample(self):
        """Implements the Scaffold sampler to identify clusters via a molecule's Bemis-Murcko scaffold."""
        scaffold_to_indices = self.scaffold_to_smiles(self.X)

        cluster_indices = np.empty(len(self.X), dtype=object)
        # give each cluster an arbitrary ID
        for cluster_id, (scaffold, indices) in enumerate(scaffold_to_indices.items()):
            if scaffold == "":
                warnings.warn(
                    f"No matching scaffold was found for the {len(indices)} " f"molecules corresponding to indices {indices}",
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
            scaffold = Scaffold.generate_bemis_murcko_scaffold(mol, self.get_config("include_chirality", False))
            scaffolds[scaffold].add(i)

        return scaffolds

    def str_to_mol(string):
        """
        Converts an InChI or SMILES string to an RDKit molecule.

        Params:
            string: The InChI or SMILES string.

        Returns:
            An RDKit molecule.
        """
        RDKIT_SMILES_PARSER_PARAMS = Chem.SmilesParserParams()
        if string.startswith("InChI"):
            mol = Chem.MolFromInchi(string, removeHs=True)
        else:
            # Set params here so we don't remove hydrogens with atom mapping
            RDKIT_SMILES_PARSER_PARAMS.removeHs = True
            mol = Chem.MolFromSmiles(string, RDKIT_SMILES_PARSER_PARAMS)

        # atom map numbers should not be present when creating scaffolds
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

        return mol

    def generate_bemis_murcko_scaffold(mol, include_chirality=False):
        """
        Compute the Bemis-Murcko scaffold for an RDKit molecule.

        Params:
            mol: A smiles string or an RDKit molecule.
            include_chirality: Whether to include chirality.

        Returns:
            Bemis-Murcko scaffold
        """
        mol = Scaffold.str_to_mol(mol) if isinstance(mol, str) else mol
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

        return scaffold
