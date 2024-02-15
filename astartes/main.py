from math import floor
from typing import Union
from warnings import warn

import numpy as np
import pandas as pd

from astartes.samplers import (
    DETERMINISTIC_EXTRAPOLATION_SAMPLERS,
    IMPLEMENTED_EXTRAPOLATION_SAMPLERS,
    IMPLEMENTED_INTERPOLATION_SAMPLERS,
)
from astartes.utils.array_type_helpers import (
    convert_to_array,
    panda_handla,
    return_helper,
)
from astartes.utils.exceptions import InvalidConfigurationError
from astartes.utils.sampler_factory import SamplerFactory
from astartes.utils.warnings import ImperfectSplittingWarning, NormalizationWarning

# define random seed
DEFAULT_RANDOM_STATE = 42


def train_val_test_split(
    X: Union[np.array, pd.DataFrame],
    y: Union[np.array, pd.Series] = None,
    labels: Union[np.array, pd.Series] = None,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    sampler: str = "random",
    random_state: int = None,
    hopts: dict = {},
    return_indices: bool = False,
):
    """Deterministic train_test_splitting of arbitrary arrays.

    Args:
        X (np.array, pd.DataFrame): Numpy array or pandas DataFrame of feature vectors.
        y (np.array, pd.Series, optional): Targets corresponding to X, must be of same size. Defaults to None.
        labels (np.array, pd.Series, optional): Labels corresponding to X, must be of same size. Defaults to None.
        train_size (float, optional): Fraction of dataset to use in training set. Defaults to 0.8.
        val_size (float, optional): Fraction of dataset to use in validation set. Defaults to 0.1.
        test_size (float, optional): Fraction of dataset to use in test set. Defaults to 0.1.
        sampler (str, optional): Sampler to use, see IMPLEMENTED_INTER/EXTRAPOLATION_SAMPLERS. Defaults to "random".
        random_state (int, optional): The random seed used throughout astartes.
        hopts (dict, optional): Hyperparameters for the sampler used above. Defaults to {}.
        return_indices (bool, optional): True to return indices of train/test after values. Defaults to False.

    Returns:
        np.array(s): X, y, and labels train/val/test data, or indices.
    """
    # special case for casting back to pandas
    output_is_pandas = panda_handla(X, y, labels)

    # now convert everything to numpy arrays for our internal algorithms
    if type(X) is not np.ndarray:
        X = convert_to_array(X, "X")
    if y is not None and type(y) is not np.ndarray:
        y = convert_to_array(y, "y")
    if labels is not None and type(labels) is not np.ndarray:
        labels = convert_to_array(labels, "labels")

    # check for consistent length after conversion
    msg = ""
    if y is not None and len(y) != len(X):
        msg += "len(y)={:d} ".format(len(y))
    if labels is not None and len(labels) != len(X):
        msg += "len(labels)={:d} ".format(len(labels))
    if msg:
        raise InvalidConfigurationError("Lengths of input arrays do not match: len(X)={:d} ".format(len(X)) + msg)

    train_size, val_size, test_size = _normalize_split_sizes(train_size, val_size, test_size)
    hopts["random_state"] = random_state if random_state is not None else DEFAULT_RANDOM_STATE
    sampler_factory = SamplerFactory(sampler)
    sampler_instance = sampler_factory.get_sampler(X, y, labels, hopts)

    if sampler in (
        *IMPLEMENTED_INTERPOLATION_SAMPLERS,
        *DETERMINISTIC_EXTRAPOLATION_SAMPLERS,
    ):
        # extrapolation samplers in DETERMINISTIC_EXTRAPOLATION_SAMPLERS do not accept the
        # random_state argument because they are entirely deterministic
        return _interpolative_sampling(
            sampler_instance,
            test_size,
            val_size,
            train_size,
            return_indices,
            output_is_pandas,
        )
    else:
        return _extrapolative_sampling(
            sampler_instance,
            test_size,
            val_size,
            train_size,
            return_indices,
            output_is_pandas,
            random_state,
        )


def train_test_split(
    X: np.array,
    y: np.array = None,
    labels: np.array = None,
    train_size: float = 0.75,
    test_size: float = None,
    sampler: str = "random",
    random_state: int = None,
    hopts: dict = {},
    return_indices: bool = False,
):
    """Deterministic train_test_splitting of arbitrary arrays.

    Args:
        X (np.array): Numpy array of feature vectors.
        y (np.array, optional): Targets corresponding to X, must be of same size. Defaults to None.
        labels (np.array, optional): Labels corresponding to X, must be of same size. Defaults to None.
        train_size (float, optional): Fraction of dataset to use in training set. Defaults to 0.75.
        test_size (float, optional): Fraction of dataset to use in test set. Defaults to None.
        sampler (str, optional): Sampler to use, see IMPLEMENTED_INTER/EXTRAPOLATION_SAMPLERS. Defaults to "random".
        random_state (int, optional): The random seed used throughout astartes.
        hopts (dict, optional): Hyperparameters for the sampler used above. Defaults to {}.
        return_indices (bool, optional): True to return indices of train/test instead of values. Defaults to False.

    Returns:
        np.array: X, y, and labels train/val/test data, or indices.
    """
    return train_val_test_split(
        X,
        y,
        labels,
        train_size,
        0,
        test_size,
        sampler,
        random_state,
        hopts,
        return_indices,
    )


def _extrapolative_sampling(
    sampler_instance,
    test_size,
    val_size,
    train_size,
    return_indices,
    output_is_pandas,
    random_state,
):
    """Helper function to perform extrapolative sampling.

    Attempts to fill train, val, and test without breaking up clusters. Prioiritizes underfilling
    test and then val and then the balance goes into train (which is why the floats are given
    in that order).

    Args:
        sampler_instance (sampler): The fit sampler instance.
        test_size (float): Fraction of data to use in test.
        val_size (float): Fraction of data to use in val.
        train_size (float): Fraction of data to use in train.
        return_indices (bool): Return indices or the arrays themselves.
        output_is_pandas (array[str] or bool): True/False if output should cast to pandas,
            data needed to perform casting if True.
        random_state (int, optional): The random state used to shuffle small clusters. Default to no shuffle.

    Returns:
        calls: return_helper
    """
    # calculate "goal" splitting sizes
    n_train_samples = floor(len(sampler_instance.X) * train_size)
    n_val_samples = floor(len(sampler_instance.X) * val_size)
    n_test_samples = floor(len(sampler_instance.X) * test_size)

    if val_size == 0:
        max_shufflable_size = min(n_train_samples, n_test_samples)
    else:
        # typically, the test set and val set are smaller than the training set
        max_shufflable_size = min(n_test_samples, n_val_samples)
    # unlike interpolative, cannot calculate n_train_samples here
    # since it will vary based on cluster_lengths

    # largest clusters must go into largest set, but smaller ones can optionally
    # be shuffled
    cluster_counter = sampler_instance.get_sorted_cluster_counter(max_shufflable_size=max_shufflable_size if random_state is not None else None)

    test_idxs, val_idxs, train_idxs = (
        np.array([], dtype=int),
        np.array([], dtype=int),
        np.array([], dtype=int),
    )
    for cluster_idx, cluster_length in cluster_counter.items():
        # assemble test first, avoid overfilling
        if (len(test_idxs) + cluster_length) <= n_test_samples:
            test_idxs = np.append(test_idxs, sampler_instance.get_sample_idxs(cluster_length))
        elif (len(val_idxs) + cluster_length) <= n_val_samples:
            val_idxs = np.append(val_idxs, sampler_instance.get_sample_idxs(cluster_length))
        else:  # then balance goes into train
            train_idxs = np.append(train_idxs, sampler_instance.get_sample_idxs(cluster_length))
    _check_actual_split(train_idxs, val_idxs, test_idxs, train_size, val_size, test_size)
    return return_helper(
        sampler_instance,
        train_idxs,
        val_idxs,
        test_idxs,
        return_indices,
        output_is_pandas,
    )


def _interpolative_sampling(
    sampler_instance,
    test_size,
    val_size,
    train_size,
    return_indices,
    output_is_pandas,
):
    """Helper function to perform interpolative sampling.

    Attempts to fill train, val, and test within rounding limits. Prioiritizes underfilling
    train and then val and then the balance goes into test (this is the opposite to extrapolative
    because these samplers move outisde-in).

    Args:
        sampler_instance (sampler): The fit sampler instance.
        test_size (float): Fraction of data to use in test.
        val_size (float): Fraction of data to use in val.
        train_size (float): Fraction of data to use in train.
        return_indices (bool): Return indices or the arrays themselves.
        output_is_pandas (array[str] or bool): True/False if output should cast to pandas,
            data needed to perform casting if True.

    Returns:
        calls: return_helper
    """
    n_train_samples = floor(len(sampler_instance.X) * train_size)
    n_val_samples = floor(len(sampler_instance.X) * val_size)
    n_test_samples = len(sampler_instance.X) - (n_train_samples + n_val_samples)

    train_idxs = sampler_instance.get_sample_idxs(n_train_samples)
    val_idxs = sampler_instance.get_sample_idxs(n_val_samples)
    test_idxs = sampler_instance.get_sample_idxs(n_test_samples)

    _check_actual_split(train_idxs, val_idxs, test_idxs, train_size, val_size, test_size)
    return return_helper(
        sampler_instance,
        train_idxs,
        val_idxs,
        test_idxs,
        return_indices,
        output_is_pandas,
    )


def _check_actual_split(
    train_idxs,
    val_idxs,
    test_idxs,
    train_size,
    val_size,
    test_size,
):
    """Check for empty sets and imperfect splits in the results.

    Args:
        train_idxs (np.array): The resulting training indices from sampling.
        val_idxs (np.array): Validation indices.
        test_idxs (np.array): Testing indices.
        train_size (float): The user-requested (or normalized) training fraction.
        val_size (float): The user-requested (or normalized) validation fraction.
        test_size (float): The user-requested (or normalized) testing fraction.

    Raises:
        InvalidConfigurationError: Raised when a set turns out empty due to configuration.
    """
    # split may have resulted in empty lists
    msg = ""
    if not len(train_idxs):
        msg += " training set empty "
    if not len(test_idxs):
        msg += " testing set empty "
    if val_size and not len(val_idxs):
        msg += " validation set empty "
    if msg:
        raise InvalidConfigurationError(
            "Provided data and requested split resulted in an empty set. "
            "Dataset may be too small or requested splits may be too large. "
            "(" + msg + ")"
        )
    # split may not match exactly what they asked for
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
    """Normalize requested inputs to between zero and one (summed) with extensive validation."""
    if not train_size and not test_size:  # neither - error
        raise InvalidConfigurationError(
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
                    raise InvalidConfigurationError("If specifying only train_size, must be float between (0, 1) (got {:.2f})".format(train_size))
                test_size = 1.0 - train_size
                out = [train_size, 0, test_size]
            else:
                if test_size >= 1.0 or test_size <= 0:
                    raise InvalidConfigurationError("If specifying only test_size, must be float between (0, 1) (got {:.2f})".format(test_size))
                train_size = 1.0 - test_size
                out = [train_size, 0, test_size]
    else:  # there is a non-zero val_size
        if val_size >= 1.0 or val_size <= 0.0:
            raise InvalidConfigurationError("val_size must be a float between (0, 1) (got {:.2f})".format(val_size))
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
            if train_size:
                if train_size >= 1.0 or train_size <= 0:
                    raise InvalidConfigurationError(
                        "If specifying val_size and only train_size, must be float between (0, 1) (got {:.2f})".format(train_size)
                    )
                test_size = 1.0 - (train_size + val_size)
                out = [train_size, val_size, test_size]
            else:
                if test_size >= 1.0 or test_size <= 0:
                    raise InvalidConfigurationError(
                        "If specifying val_size and only test_size, must be float between (0, 1) (got {:.2f})".format(test_size)
                    )
                train_size = 1.0 - (test_size + val_size)
                out = [train_size, val_size, test_size]

    return (*out,)
