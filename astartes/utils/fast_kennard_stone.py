import numpy as np
from scipy.spatial.distance import pdist, squareform


def fast_kennard_stone(X, metric):
    n_samples = len(X)

    if metric == "precomputed":
        ks_distance = X
    else:
        X_dist = pdist(X, metric=metric)
        ks_distance = squareform(X_dist)
    # when searching for max distance, disregard self
    np.fill_diagonal(ks_distance, -np.inf)

    # get the row/col of maximum (greatest distance)
    max_idx = np.nanargmax(ks_distance)
    max_coords = np.unravel_index(max_idx, ks_distance.shape)

    # list of indices which have been selected
    # - used to mask ks_distance
    # - also tracks order of kennard-stone selection
    already_selected = list(max_coords)

    # minimum distance of all unselected samples to the two selected samples
    min_distances = np.min(ks_distance[:, already_selected], axis=1)
    for _ in range(n_samples - 2):
        # find the next sample with the largest minimum distance to any sample already selected
        already_selected.append(
            np.argmax(min_distances),
        )
        # get minimum distance of unselected samples to that sample only
        new_distances = np.min(ks_distance[:, [already_selected[-1]]], axis=1)
        min_distances = np.minimum(min_distances, new_distances)

    return np.array(already_selected, dtype=int)
