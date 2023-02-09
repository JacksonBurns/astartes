from difflib import get_close_matches
from math import floor

import numpy as np

from astartes.samplers import (
    Random,
    KennardStone,
)
from astartes.utils.exceptions import (
    InvalidAstartesConfigurationError,
)

from astartes.samplers import (
    IMPLEMENTED_UNSUPERVISED_SAMPLERS,
    IMPLEMENTED_SUPERVISED_SAMPLERS,
)


def train_test_split(
    X: np.array,
    y: np.array = None,
    test_size: float = None,
    train_size: float = 0.75,
    sampler: str = "random",
    hopts: dict = {},
    labels: np.array = None,
):
    """Deterministic train_test_splitting of arbitrary matrices.

    Args:
        X (np.array): Features.
        y (np.array, optional): Targets corresponding to features, must be of same size. Defaults to None.
        test_size (float, optional): Fraction of dataset to use in test set. Defaults to None.
        train_size (float, optional): Fraction of dataset to use in training (test+train~1). Defaults to 0.75.
        sampler (str, optional): Sampler to use, see IMPLEMENTED_UN/SUPERVISED_SAMPLERS. Defaults to "random".
        hopts (dict, optional): Hyperparameters for the sampler used above. Defaults to {}.
        labels (np.array, optional): Labels for supervised sampling. Defaults to None.

    Raises:
        InvalidAstartesConfigurationError: Raised when sampler is configured incorrectly.
        NotImplementedError: Raised when an invalid sampler name is used.

    Returns:
        np.array: Training and test data split into arrays.
    """
    # validationm for supervised sampling
    if sampler in IMPLEMENTED_SUPERVISED_SAMPLERS:
        msg = None
        if labels is None:  # labeled samplers need labels (but not vice versa!)
            msg = """Sampler {:s} requires labeled data, but labels were not provided.""".format(
                sampler
            )
        elif (
            np.size(X)[0] != np.size(labels)[0]
        ):  # size of data must match number of labels
            msg = """First dimension of X and labels must match (got X dim={:d} and labels dim={:d})""".format(
                np.size(X)[0],
                np.size(labels)[0],
            )
        if msg:
            raise InvalidAstartesConfigurationError(msg)
    if sampler == "random":
        sampler = Random(hopts)
    elif sampler == "kennard_stone":
        sampler = KennardStone(hopts)
    else:
        possiblity = get_close_matches(
            sampler,
            IMPLEMENTED_SUPERVISED_SAMPLERS + IMPLEMENTED_UNSUPERVISED_SAMPLERS,
            n=1,
        )
        addendum = (
            " Did you mean '{:s}'?".format(possiblity[0])
            if possiblity
            else " Try help(train_test_split)."
        )
        raise NotImplementedError(
            "Sampler {:s} has not been implemented.".format(sampler) + addendum
        )

    sampler.populate(X, y)

    if test_size is not None:
        train_size = 1.0 - test_size

    n_train_samples = floor(len(X) * train_size)
    train_idxs = sampler.get_sample_idxs(n_train_samples)
    test_idxs = sampler.get_sample_idxs(len(X) - n_train_samples)
    X_train = np.array([X[i] for i in train_idxs], dtype=object)
    X_test = np.array([X[i] for i in test_idxs], dtype=object)
    if y is None:
        return X_train, X_test
    else:
        y_train = np.array([y[i] for i in train_idxs], dtype=object)
        y_test = np.array([y[i] for i in test_idxs], dtype=object)
        return X_train, X_test, y_train, y_test
