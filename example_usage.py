from astartes import train_test_split
from astartes.samplers import KennardStone

import numpy as np


X = np.array(list(range(10))).reshape(-1, 1)
y = np.array(list(range(10, 20))).reshape(-1, 1)


ks = KennardStone(configs={'train_size': 0.8})

ks.populate(X, y)

a = ks.get_samples(2)
print(a)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    splitter='kennard_stone',
)

print(X_train, X_test, y_train, y_test)
