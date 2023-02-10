from difflib import get_close_matches
from math import floor

import numpy as np

from astartes.samplers import (
    Random,
    KennardStone,
    KMeans,
)
from astartes.utils.exceptions import (
    InvalidAstartesConfigurationError,
)

from astartes.samplers import (
    IMPLEMENTED_INTERPOLATION_SAMPLERS,
    IMPLEMENTED_EXTRAPOLATION_SAMPLERS,
    ALL_SAMPLERS,
)


def train_test_split(
    X: np.array,
    y: np.array = None,
    labels: np.array = None,
    test_size: float = None,
    train_size: float = 0.75,
    sampler: str = "random",
    hopts: dict = {},
    return_indices: bool = False,
):
    """Deterministic train_test_splitting of arbitrary matrices.

    Args:
        X (np.array): Features.
        y (np.array, optional): Targets corresponding to features, must be of same size. Defaults to None.
        test_size (float, optional): Fraction of dataset to use in test set. Defaults to None.
        train_size (float, optional): Fraction of dataset to use in training (test+train~1). Defaults to 0.75.
        sampler (str, optional): Sampler to use, see IMPLEMENTED_INTER/EXTRAPOLATION_SAMPLERS. Defaults to "random".
        hopts (dict, optional): Hyperparameters for the sampler used above. Defaults to {}.
        labels (np.array, optional): Labels for supervised sampling. Defaults to None.

    Raises:
        InvalidAstartesConfigurationError: Raised when sampler is configured incorrectly.
        NotImplementedError: Raised when an invalid sampler name is used.

    Returns:
        np.array: Training and test data split into arrays.
    """
    if sampler not in ALL_SAMPLERS:
        possiblity = get_close_matches(
            sampler,
            IMPLEMENTED_EXTRAPOLATION_SAMPLERS + IMPLEMENTED_INTERPOLATION_SAMPLERS,
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

    if sampler in IMPLEMENTED_EXTRAPOLATION_SAMPLERS:
        return _extrapolative_sampling(
            X,
            y,
            labels,
            test_size,
            train_size,
            sampler,
            hopts,
            return_indices,
        )
    else:
        return _interpolative_sampling(
            X,
            y,
            labels,
            test_size,
            train_size,
            sampler,
            hopts,
            return_indices,
        )


def _extrapolative_sampling(
    X: np.array,
    y: np.array = None,
    labels: np.array = None,
    test_size: float = None,
    train_size: float = 0.75,
    sampler: str = "random",
    hopts: dict = {},
    return_indices: bool = False,
):
    # validation for extrapolative sampling
    msg = None
    if labels is None:  # labeled samplers need labels (but not vice versa!)
        msg = """Sampler {:s} requires labeled data, but labels were not provided.""".format(
            sampler
        )
    elif len(X) != np.size(labels):  # size of data must match number of labels
        msg = """First dimension of X and labels must match (got X dim={:d} and labels dim={:d})""".format(
            len(X),
            np.size(labels),
        )
    if msg:
        raise InvalidAstartesConfigurationError(msg)

    sampler_class = None
    if sampler == "kmeans":
        sampler_class = KMeans

    sampler_instance = sampler_class(X, y, labels, hopts)

    if test_size is None:
        test_size = 1.0 - train_size

    n_test_samples = floor(len(X) * test_size)

    # can't break up clusters
    # warn the user that the exact split they asked for was not possible
    # this is what they get isntead
    # we exoect that supervised samplers will return lists of tuples
    # where tuple(1) is the cluster id

    cluster_counter = sampler_instance.get_cluster_counter()
    test_idxs, train_idxs = [], []
    print(cluster_counter)
    for cluster_idx, n_samples in cluster_counter.items():
        print(n_samples)
        # assemble test first
        if (len(test_idxs) + n_samples) < n_test_samples:  # will not overfill
            test_idxs.append(sampler_instance.get_sample_idxs(n_samples))
    else:  # then balance goes into train
        train_idxs.append(sampler_instance.get_sample_idxs(n_samples))
    # TODO: get the clusters in order, get the other array s(abstract into a helper for both types of splitting) and then abstract the actual result checker and clal it from both


def _interpolative_sampling(
    X: np.array,
    y: np.array = None,
    labels: np.array = None,
    test_size: float = None,
    train_size: float = 0.75,
    sampler: str = "random",
    hopts: dict = {},
    return_indices: bool = False,
):
    sampler_class = None
    if sampler == "random":
        sampler_class = Random
    elif sampler == "kennard_stone":
        sampler_class = KennardStone

    sampler_instance = sampler_class(X, y, labels, hopts)

    if test_size is not None:
        train_size = 1.0 - test_size

    n_train_samples = floor(len(X) * train_size)
    # TODO: warm the user about what the actual split came out to be

    train_idxs = sampler_instance.get_sample_idxs(n_train_samples)
    test_idxs = sampler_instance.get_sample_idxs(len(X) - n_train_samples)
    if return_indices:
        return train_idxs, test_idxs
    else:
        out = []
        X_train = np.array([X[i] for i in train_idxs], dtype=object)
        out.append(X_train)
        X_test = np.array([X[i] for i in test_idxs], dtype=object)
        out.append(X_test)

        if y is not None:
            y_train = np.array([y[i] for i in train_idxs], dtype=object)
            out.append(y_train)
            y_test = np.array([y[i] for i in test_idxs], dtype=object)
            out.append(y_test)
        if labels is not None:
            labels_train = np.array([labels[i] for i in train_idxs], dtype=object)
            out.append(labels_train)
            labels_test = np.array([labels[i] for i in test_idxs], dtype=object)
            out.append(labels_test)

        return (*out,)
