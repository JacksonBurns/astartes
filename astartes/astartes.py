import numpy as np
from math import floor
from astartes.samplers import (
    Random,
    KennardStone,
)

IMPLEMENTED_SAMPLERS = [
    'random',
    'kennard_stone',
]


def train_test_split(
    X: np.array,
    y: np.array = None,
    test_size: float = None,
    train_size: float = 0.75,
    splitter: str = 'random',
    hopts: dict = {},
):
    if splitter == 'random':
        sampler = Random(hopts)
    elif splitter == 'kennard_stone':
        sampler = KennardStone(hopts)
    else:
        raise NotImplementedError(
            f'Splitter "{splitter}" has not been implemented.'
        )

    sampler.populate(X, y)

    if test_size is not None:
        train_size = 1.0 - test_size

    n_train_samples = floor(len(X) * train_size)
    train_idxs = sampler.get_sample_idxs(n_train_samples)
    test_idxs = sampler.get_sample_idxs(len(X)-n_train_samples)
    X_train = np.array([X[i] for i in train_idxs])
    X_test = np.array([X[i] for i in test_idxs])
    if y is None:
        return X_train, X_test
    else:
        y_train = np.array([y[i] for i in train_idxs])
        y_test = np.array([y[i] for i in test_idxs])
        return X_train, X_test, y_train, y_test
