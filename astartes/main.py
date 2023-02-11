from difflib import get_close_matches
from math import floor
from warnings import warn

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
    sampler = sampler.lower()
    if sampler == "random":
        sampler_class = Random
    elif sampler == "kennard_stone":
        sampler_class = KennardStone
    elif sampler == "kmeans":
        sampler_class = KMeans
    else:
        possiblity = get_close_matches(
            sampler,
            ALL_SAMPLERS,
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

    sampler_instance = sampler_class(X, y, labels, hopts)

    if test_size is not None:
        train_size = 1.0 - test_size

    if sampler in IMPLEMENTED_EXTRAPOLATION_SAMPLERS:
        return _extrapolative_sampling(
            sampler_instance,
            test_size,
            train_size,
            return_indices,
        )
    else:
        return _interpolative_sampling(
            sampler_instance,
            test_size,
            train_size,
            return_indices,
        )


def _extrapolative_sampling(
    sampler_instance,
    test_size,
    train_size,
    return_indices,
):
    # validation for extrapolative sampling
    n_test_samples = floor(len(sampler_instance.X) * test_size)

    # can't break up clusters
    cluster_counter = sampler_instance.get_sorted_cluster_counter()
    test_idxs, train_idxs = np.array([], dtype=int), np.array([], dtype=int)
    for cluster_idx, cluster_length in cluster_counter.items():
        # assemble test first
        print(cluster_idx, len(test_idxs), cluster_length, n_test_samples)
        if (len(test_idxs) + cluster_length) <= n_test_samples:  # will not overfill
            test_idxs = np.append(
                test_idxs, sampler_instance.get_sample_idxs(cluster_length)
            )
        else:  # then balance goes into train
            train_idxs = np.append(
                train_idxs, sampler_instance.get_sample_idxs(cluster_length)
            )
        print(test_idxs, train_idxs)
    _check_actual_split(train_idxs, test_idxs, train_size, test_size)
    return _return_helper(sampler_instance, train_idxs, test_idxs, return_indices)


def _interpolative_sampling(
    sampler_instance,
    test_size,
    train_size,
    return_indices,
):

    n_train_samples = floor(len(sampler_instance.X) * train_size)

    train_idxs = sampler_instance.get_sample_idxs(n_train_samples)
    test_idxs = sampler_instance.get_sample_idxs(
        len(sampler_instance.X) - n_train_samples
    )
    _check_actual_split(train_idxs, test_idxs, train_size, test_size)
    return _return_helper(sampler_instance, train_idxs, test_idxs, return_indices)


def _return_helper(
    sampler_instance,
    train_idxs,
    test_idxs,
    return_indices,
):
    if return_indices:
        return train_idxs, test_idxs
    out = []
    X_train = sampler_instance.X[train_idxs]
    out.append(X_train)
    X_test = sampler_instance.X[test_idxs]
    out.append(X_test)

    if sampler_instance.y is not None:
        y_train = np.array([sampler_instance.y[i] for i in train_idxs], dtype=object)
        out.append(y_train)
        y_test = np.array([sampler_instance.y[i] for i in test_idxs], dtype=object)
        out.append(y_test)
    if sampler_instance.labels is not None:
        labels_train = np.array(
            [sampler_instance.labels[i] for i in train_idxs], dtype=object
        )
        out.append(labels_train)
        labels_test = np.array(
            [sampler_instance.labels[i] for i in test_idxs], dtype=object
        )
        out.append(labels_test)
    if len(sampler_instance.get_clusters()):  # true when the list has been filled
        clusters_train = np.array(
            [sampler_instance.get_clusters()[i] for i in train_idxs], dtype=object
        )
        out.append(clusters_train)
        clusters_test = np.array(
            [sampler_instance.get_clusters()[i] for i in test_idxs], dtype=object
        )
        out.append(clusters_test)

    return (*out,)


def _check_actual_split(train_idxs, test_idxs, train_size, test_size):
    # print(train_idxs, test_idxs, train_size, test_size)
    actual_train_size = round(len(train_idxs) / (len(test_idxs) + len(train_idxs)), 3)
    requested_train_size = round(train_size, 3)
    actual_test_size = round(1.0 - actual_train_size, 3)
    requested_test_size = round(test_size, 3)
    msg = ""
    if actual_train_size != requested_train_size:
        msg += "Requested train size of {:.2f}, got {:.2f}. ".format(
            requested_train_size,
            actual_train_size,
        )
    if actual_test_size != requested_test_size:
        msg += "Requested test size of {:.2f}, got {:.2f}. ".format(
            requested_test_size,
            actual_test_size,
        )
    if msg:
        warn(
            "Actual train/test split differs from requested size. " + msg,
            RuntimeWarning,
        )
