import numpy as np
from astartes.samplers import (
    random,
    kennard_stone,
)


def train_test_split(
    X: np.array,
    y: np.array = None,
    test_size: float = 0.25,
    train_size: float = 0.75,
    splitter: str = 'random',
    hopts: dict = {},
):
    if splitter == 'random':
        return random(
            X,
            y=y,
            test_size=test_size,
            train_size=train_size,
            shuffle=hopts.get("shuffle", False),
            random_state=hopts.get("random_state", None),
            stratify=hopts.get("stratify", None),
        )
    elif splitter == 'kennard_stone':
        return kennard_stone(
            X,
            y=y,
            test_size=test_size,
            train_size=train_size,
        )
    else:
        raise NotImplementedError(
            f'Splitter "{splitter}" has not been implemented.'
        )
