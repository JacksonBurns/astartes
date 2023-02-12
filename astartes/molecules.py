from typing import List

import numpy as np

from astartes.utils.exceptions import MoleculesNotInstalledError

try:
    from aimsim.chemical_datastructures import Molecule
except ImportError:
    raise MoleculesNotInstalledError(
        """To use molecule featurizer, install astartes with pip install astartes[molecules]."""
    )


from astartes import train_test_split


def train_test_split_molecules(
    smiles: List[str],
    y: np.array = None,
    test_size: float = 0.25,
    train_size: float = 0.75,
    sampler: str = "random",
    fingerprint: str = "morgan_fingerprint",
    return_as: str = "fprint",
    hopts: dict = {},
    fprints_hopts: dict = {},
):
    # turn the smiles into an input X
    X = []
    for smile in smiles:
        mol = Molecule(mol_smiles=smile)
        mol.descriptor.make_fingerprint(
            mol.mol_graph,
            fingerprint,
            fingerprint_params=fprints_hopts,
        )
        X.append(mol.descriptor.to_numpy())
    X = np.array(X)
    # call train test split with this input
    return train_test_split(
        X,
        y=y,
        test_size=test_size,
        train_size=train_size,
        sampler=sampler,
        hopts=hopts,
    )
