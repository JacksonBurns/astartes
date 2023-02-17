from math import floor
from warnings import warn

import numpy as np

from astartes.samplers import (
    IMPLEMENTED_EXTRAPOLATION_SAMPLERS,
    IMPLEMENTED_INTERPOLATION_SAMPLERS,
)
from astartes.utils.sampler_factory import SamplerFactory
from astartes.utils.warnings import ImperfectSplittingWarning, NormalizationWarning


def train_val_test_split(
    X: np.array,
    y: np.array = None,
    labels: np.array = None,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    sampler: str = "random",
    hopts: dict = {},
    return_indices: bool = False,
):
    train_size, val_size, test_size = _normalize_split_sizes(
        train_size, val_size, test_size
    )
    sampler_factory = SamplerFactory(sampler)
    sampler_instance = sampler_factory.get_sampler(X, y, labels, hopts)

    if sampler in IMPLEMENTED_EXTRAPOLATION_SAMPLERS:
        return _extrapolative_sampling(
            sampler_instance,
            test_size,
            val_size,
            train_size,
            return_indices,
        )
    else:
        return _interpolative_sampling(
            sampler_instance,
            test_size,
            val_size,
            train_size,
            return_indices,
        )


def train_test_split(
    X: np.array,
    y: np.array = None,
    labels: np.array = None,
    train_size: float = 0.75,
    test_size: float = None,
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
        NotImplementedError: Raised when an invalid sampler name is used.

    Returns:
        np.array: Training and test data split into arrays.
    """
    return train_val_test_split(
        X, y, labels, train_size, 0, test_size, sampler, hopts, return_indices
    )


def _extrapolative_sampling(
    sampler_instance,
    test_size,
    val_size,
    train_size,
    return_indices,
):
    # validation for extrapolative sampling
    n_test_samples = floor(len(sampler_instance.X) * test_size)
    n_val_samples = floor(len(sampler_instance.X) * val_size)

    # can't break up clusters
    cluster_counter = sampler_instance.get_sorted_cluster_counter()
    test_idxs, val_idxs, train_idxs = (
        np.array([], dtype=int),
        np.array([], dtype=int),
        np.array([], dtype=int),
    )
    for cluster_idx, cluster_length in cluster_counter.items():
        # assemble test first
        if (len(test_idxs) + cluster_length) <= n_test_samples:  # will not overfill
            test_idxs = np.append(
                test_idxs, sampler_instance.get_sample_idxs(cluster_length)
            )
        if (len(val_idxs) + cluster_length) <= n_val_samples:
            val_idxs = np.append(
                val_idxs, sampler_instance.get_sample_idxs(cluster_length)
            )
        else:  # then balance goes into train
            train_idxs = np.append(
                train_idxs, sampler_instance.get_sample_idxs(cluster_length)
            )
    _check_actual_split(
        train_idxs, val_idxs, test_idxs, train_size, val_size, test_size
    )
    return _return_helper(
        sampler_instance, train_idxs, val_idxs, test_idxs, return_indices
    )


def _interpolative_sampling(
    sampler_instance,
    test_size,
    val_size,
    train_size,
    return_indices,
):
    # build test first
    n_test_samples = floor(len(sampler_instance.X) * test_size)
    n_val_samples = floor(len(sampler_instance.X) * val_size)
    n_train_samples = len(sampler_instance.X) - (n_test_samples + n_val_samples)

    test_idxs = sampler_instance.get_sample_idxs(n_test_samples)
    val_idxs = sampler_instance.get_sample_idxs(n_val_samples)
    train_idxs = sampler_instance.get_sample_idxs(n_train_samples)

    _check_actual_split(
        train_idxs, val_idxs, test_idxs, train_size, val_size, test_size
    )
    return _return_helper(
        sampler_instance, train_idxs, val_idxs, test_idxs, return_indices
    )


def _return_helper(
    sampler_instance,
    train_idxs,
    val_idxs,
    test_idxs,
    return_indices,
):
    if return_indices:
        if val_idxs:
            return train_idxs, val_idxs, test_idxs
        else:
            return train_idxs, test_idxs
    out = []
    X_train = sampler_instance.X[train_idxs]
    out.append(X_train)
    if val_idxs:
        X_val = sampler_instance.X[val_idxs]
        out.append(X_val)
    X_test = sampler_instance.X[test_idxs]
    out.append(X_test)

    if sampler_instance.y is not None:
        y_train = sampler_instance.y[train_idxs]
        out.append(y_train)
        if val_idxs:
            y_val = sampler_instance.y[val_idxs]
            out.append(y_val)
        y_test = sampler_instance.y[test_idxs]
        out.append(y_test)
    if sampler_instance.labels is not None:
        labels_train = sampler_instance.labels[train_idxs]
        out.append(labels_train)
        if val_idxs:
            labels_val = sampler_instance.labels[val_idxs]
            out.append(labels_val)
        labels_test = sampler_instance.labels[test_idxs]
        out.append(labels_test)
    if len(sampler_instance.get_clusters()):  # true when the list has been filled
        clusters_train = sampler_instance.get_clusters()[train_idxs]
        out.append(clusters_train)
        if val_idxs:
            clusters_val = sampler_instance.get_clusters()[val_idxs]
            out.append(clusters_val)
        clusters_test = sampler_instance.get_clusters()[test_idxs]
        out.append(clusters_test)

    return (*out,)


def _check_actual_split(
    train_idxs,
    val_idxs,
    test_idxs,
    train_size,
    val_size,
    test_size,
):
    total_size = len(test_idxs) + len(val_idxs) + len(train_idxs)

    actual_train_size = round(len(train_idxs) / total_size, 2)
    requested_train_size = round(train_size, 2)

    actual_val_size = round(len(val_idxs) / total_size, 2)
    requested_val_size = round(val_size, 2)

    actual_test_size = round(len(test_idxs) / total_size, 2)
    requested_test_size = round(test_size, 2)

    msg = ""
    if actual_train_size != requested_train_size:
        msg += "Requested train size of {:.2f}, got {:.2f}. ".format(
            requested_train_size,
            actual_train_size,
        )
    if actual_val_size != requested_val_size:
        msg += "Requested validation size of {:.2f}, got {:.2f}. ".format(
            requested_test_size,
            actual_test_size,
        )
    if actual_test_size != requested_test_size:
        msg += "Requested test size of {:.2f}, got {:.2f}. ".format(
            requested_test_size,
            actual_test_size,
        )
    if msg:
        warn(
            "Actual train/test split differs from requested size. " + msg,
            ImperfectSplittingWarning,
        )


def _normalize_split_sizes(train_size, val_size, test_size):
    """Normalize requested inputs to between zero and one (summed)."""
    if not train_size and not test_size:  # neither - error
        raise RuntimeError(
            "train_size or test_size must be nonzero (val_size will default to 0.0).\n"
            "(got val_size={:s} test_size={:s} train_size={:s})".format(
                repr(val_size),
                repr(test_size),
                repr(train_size),
            )
        )
    out = []
    if not val_size:  # doing train_test_split
        if train_size and test_size:  # both - normalize
            if train_size + test_size != 1.0:
                out_train_size = train_size / (train_size + test_size)
                out_test_size = 1.0 - out_train_size
                warn(
                    "Requested train/test split ({:.2f}, {:.2f}) do not sum to 1.0,"
                    " normalizing to train={:.2f}, test={:.2f}.".format(
                        train_size,
                        test_size,
                        out_train_size,
                        out_test_size,
                    ),
                    NormalizationWarning,
                )
                out = [out_train_size, 0.0, out_test_size]
            else:
                out = [train_size, val_size, test_size]
        else:  # one or the other - only allow floats [0, 1), then calculate
            if train_size:
                if train_size >= 1.0 or train_size <= 0:
                    raise RuntimeError(
                        "If specifying only train_size, must be float between (0, 1) (got {:.2f})".format(
                            train_size
                        )
                    )
                test_size = 1.0 - train_size
                out = [train_size, 0, test_size]
            else:
                if test_size >= 1.0 or test_size <= 0:
                    raise RuntimeError(
                        "If specifying only test_size, must be float between (0, 1) (got {:.2f})".format(
                            test_size
                        )
                    )
                train_size = 1.0 - test_size
                out = [train_size, 0, test_size]
    else:
        if train_size and test_size:  # all three - normalize
            if train_size + test_size + val_size != 1.0:
                normalization = train_size + test_size + val_size
                out_train_size = train_size / normalization
                out_test_size = test_size / normalization
                out_val_size = val_size / normalization
                warn(
                    "Requested train/val/test split ({:.2f}, {:.2f}, {:.2f}) do not sum to 1.0,"
                    " normalizing to train={:.2f}, val={:.2f}, test={:.2f}.".format(
                        train_size,
                        val_size,
                        test_size,
                        out_train_size,
                        out_val_size,
                        out_test_size,
                    ),
                    NormalizationWarning,
                )
                out = [out_train_size, val_size, out_test_size]
            else:
                out = [train_size, val_size, test_size]
        else:  # one or the other - only allow floats [0, 1), then calculate
            if val_size >= 1.0 or val_size <= 0.0:
                raise RuntimeError(
                    "val_size must be a float between (0, 1) (for {:.2f})".format(
                        val_size
                    )
                )
            if train_size:
                if train_size >= 1.0 or train_size <= 0:
                    raise RuntimeError(
                        "If specifying val_size and only train_size, must be float between (0, 1) (got {:.2f})".format(
                            train_size
                        )
                    )
                test_size = 1.0 - (train_size + val_size)
                out = [train_size, val_size, test_size]
            else:
                if test_size >= 1.0 or test_size <= 0:
                    raise RuntimeError(
                        "If specifying val_size and only test_size, must be float between (0, 1) (got {:.2f})".format(
                            test_size
                        )
                    )
                train_size = 1.0 - (test_size + val_size)
                out = [train_size, val_size, test_size]

    return (*out,)
