from typing import List

import numpy as np

from astartes.utils.exceptions import MoleculesNotInstalledError

try:
    from aimsim.chemical_datastructures import Molecule
except ImportError:  # pragma: no cover
    raise MoleculesNotInstalledError(
        """To use molecule featurizer, install astartes with pip install astartes[molecules]."""
    )


from astartes import train_test_split, train_val_test_split


def train_val_test_split_molecules(
    smiles: List[str],
    y: np.array = None,
    labels: np.array = None,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    sampler: str = "random",
    hopts: dict = {},
    fingerprint: str = "morgan_fingerprint",
    return_as: str = "fprint",
    fprints_hopts: dict = {},
):
    X = _featurize(smiles, fingerprint, fprints_hopts)
    return train_val_test_split(
        X,
        y=y,
        labels=labels,
        test_size=test_size,
        val_size=val_size,
        train_size=train_size,
        sampler=sampler,
        hopts=hopts,
    )


def train_test_split_molecules(
    smiles: List[str],
    y: np.array = None,
    labels: np.array = None,
    train_size: float = 0.75,
    test_size: float = None,
    sampler: str = "random",
    hopts: dict = {},
    fingerprint: str = "morgan_fingerprint",
    return_as: str = "fprint",
    fprints_hopts: dict = {},
):
    # turn the smiles into an input X
    X = _featurize(smiles, fingerprint, fprints_hopts)

    # call train test split with this input
    return train_test_split(
        X,
        y=y,
        labels=labels,
        test_size=test_size,
        train_size=train_size,
        sampler=sampler,
        hopts=hopts,
    )


def _featurize(smiles, fingerprint, fprints_hopts):
    X = []
    for smile in smiles:
        mol = Molecule(mol_smiles=smile)
        mol.descriptor.make_fingerprint(
            mol.mol_graph,
            fingerprint,
            fingerprint_params=fprints_hopts,
        )
        X.append(mol.descriptor.to_numpy())
    return np.array(X)
