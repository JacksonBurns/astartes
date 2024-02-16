import warnings

import numpy as np

from astartes.utils.exceptions import MoleculesNotInstalledError

Molecule = None
try:
    """
    aimsim depends on sklearn_extra, which uses a version checking technique that is due to
    be deprecated in a version of Python after 3.11, so it is throwing a deprecation warning
    We ignore this warning since we can't do anything about it (sklearn_extra seems to be
    abandonware) and in the future it will become an error that we can deal with.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        from aimsim.chemical_datastructures import Molecule
        from aimsim.exceptions import LoadingError
except ImportError:  # pragma: no cover
    pass  # Raise an error later


def featurize_molecules(molecules, fingerprint, fprints_hopts):
    """Call AIMSim's Molecule to featurize the molecules according to the arguments.

    Args:
        molecules (np.array): SMILES strings or RDKit molecule objects.
        fingerprint (str): The molecular fingerprint to be used.
        fprints_hopts (dict): Hyperparameters for AIMSim.

    Returns:
        np.array: X array (featurized molecules)
    """
    if Molecule is None:
        raise MoleculesNotInstalledError(
            """To use molecule featurizer, install astartes with pip install astartes[molecules] or conda install astartes aimsim."""
        )
    X = []
    for molecule in molecules:
        try:
            if type(molecule) in (np.str_, str):
                mol = Molecule(mol_smiles=molecule)
            else:
                mol = Molecule(mol_graph=molecule)
        except LoadingError as le:
            raise RuntimeError(
                "Unable to featurize molecules using '{:s}' with this configuration: fprint_hopts={:s}"
                "\nCheck terminal output for messages from the RDkit logger. ".format(fingerprint, repr(fprints_hopts))
            ) from le
        mol.descriptor.make_fingerprint(
            mol.mol_graph,
            fingerprint,
            fingerprint_params=fprints_hopts,
        )
        X.append(mol.descriptor.to_numpy())
    return np.array(X)
