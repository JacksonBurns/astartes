import numpy as np
from math import floor
from astartes.samplers import (
    Random,
    KennardStone,
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
    # if
    if sampler == "random":
        sampler = Random(hopts)
    elif sampler == "kennard_stone":
        sampler = KennardStone(hopts)
    else:
        raise NotImplementedError(f'Sampler "{sampler}" has not been implemented.')

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
