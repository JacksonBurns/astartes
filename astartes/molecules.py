import numpy as np

# at this point we have successfully verified that rdkit is installed, so we can do this:
from rdkit.rdBase import SeedRandomNumberGenerator

from astartes import train_test_split, train_val_test_split
from astartes.main import DEFAULT_RANDOM_STATE
from astartes.utils.aimsim_featurizer import featurize_molecules

SeedRandomNumberGenerator(DEFAULT_RANDOM_STATE)


def train_val_test_split_molecules(
    molecules: np.array,
    y: np.array = None,
    labels: np.array = None,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    sampler: str = "random",
    random_state: int = None,
    hopts: dict = {},
    fingerprint: str = "morgan_fingerprint",
    fprints_hopts: dict = {},
    return_indices: bool = False,
):
    """Deterministic train_test_splitting of molecules (SMILES strings or RDKit objects).

    Args:
        molecules (np.array): List of SMILES strings or RDKit molecule objects representing molecules or reactions.
        y (np.array, optional): Targets corresponding to SMILES, must be of same size. Defaults to None.
        labels (np.array, optional): Labels corresponding to SMILES, must be of same size. Defaults to None.
        train_size (float, optional): Fraction of dataset to use in training set. Defaults to 0.8.
        val_size (float, optional): Fraction of dataset to use in validation set. Defaults to 0.1.
        test_size (float, optional): Fraction of dataset to use in test set. Defaults to 0.1.
        sampler (str, optional): Sampler to use, see IMPLEMENTED_INTER/EXTRAPOLATION_SAMPLERS. Defaults to "random".
        random_state (int, optional): The random seed used throughout astartes. Defaults to 42.
        hopts (dict, optional): Hyperparameters for the sampler used above. Defaults to {}.
        fingerprint (str, optional): Molecular fingerprint to be used from AIMSim. Defaults to "morgan_fingerprint".
        fprints_hopts (dict, optional): Hyperparameters for AIMSim featurization. Defaults to {}.
        return_indices (bool, optional): True to return indices of train/test after the values. Defaults to False.

    Returns:
        np.array: X, y, and labels train/val/test data, or indices.
    """
    if sampler == "scaffold":
        X = molecules
    else:
        X = featurize_molecules(molecules, fingerprint, fprints_hopts)
    return train_val_test_split(
        X,
        y=y,
        labels=labels,
        test_size=test_size,
        val_size=val_size,
        train_size=train_size,
        sampler=sampler,
        random_state=random_state,
        hopts=hopts,
        return_indices=return_indices,
    )


def train_test_split_molecules(
    molecules: np.array,
    y: np.array = None,
    labels: np.array = None,
    train_size: float = 0.75,
    test_size: float = None,
    sampler: str = "random",
    random_state: int = None,
    hopts: dict = {},
    fingerprint: str = "morgan_fingerprint",
    fprints_hopts: dict = {},
    return_indices: bool = False,
):
    """Deterministic train/test splitting of molecules (SMILES strings or RDKit objects).

    Args:
        molecules (np.array): List of SMILES strings or RDKit molecule objects representing molecules or reactions.
        y (np.array, optional): Targets corresponding to SMILES, must be of same size. Defaults to None.
        labels (np.array, optional): Labels corresponding to SMILES, must be of same size. Defaults to None.
        train_size (float, optional): Fraction of dataset to use in training (test+train~1). Defaults to 0.75.
        test_size (float, optional): Fraction of dataset to use in test set. Defaults to None.
        sampler (str, optional): Sampler to use, see IMPLEMENTED_INTER/EXTRAPOLATION_SAMPLERS. Defaults to "random".
        random_state (int, optional): The random seed used throughout astartes. Defaults to None.
        hopts (dict, optional): Hyperparameters for the sampler used above. Defaults to {}.
        fingerprint (str, optional): Molecular fingerprint to be used from AIMSim. Defaults to "morgan_fingerprint".
        fprints_hopts (dict, optional): Hyperparameters for AIMSim featurization. Defaults to {}.
        return_indices (bool, optional): True to return indices of train/test after the values. Defaults to False.

    Returns:
        np.array: X, y, and labels train/test data, or indices.
    """
    # turn the smiles into an input X
    if sampler == "scaffold":
        X = molecules
    else:
        X = featurize_molecules(molecules, fingerprint, fprints_hopts)

    # call train test split with this input
    return train_test_split(
        X,
        y=y,
        labels=labels,
        test_size=test_size,
        train_size=train_size,
        sampler=sampler,
        random_state=random_state,
        hopts=hopts,
        return_indices=return_indices,
    )
