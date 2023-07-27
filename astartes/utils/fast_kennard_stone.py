import numpy as np


def fast_kennard_stone(ks_distance: np.ndarray) -> np.ndarray:
    """Implements the Kennard-Stone algorithm

    Args:
        ks_distance (np.ndarray): Distance matrix

    Returns:
        np.ndarray: Indices in order of Kennard-Stone selection
    """
    n_samples = len(ks_distance)

    # when searching for max distance, disregard self
    np.fill_diagonal(ks_distance, -np.inf)

    # get the row/col of maximum (greatest distance)
    max_idx = np.nanargmax(ks_distance)
    max_coords = np.unravel_index(max_idx, ks_distance.shape)

    # list of indices which have been selected
    # - used to mask ks_distance
    # - also tracks order of kennard-stone selection
    already_selected = np.empty(n_samples, dtype=int)
    already_selected[0] = max_coords[0]
    already_selected[1] = max_coords[1]

    # minimum distance of all unselected samples to the two selected samples
    min_distances = np.min(ks_distance[:, max_coords], axis=1)
    for i in range(2, n_samples):
        # find the next sample with the largest minimum distance to any sample already selected
        already_selected[i] = np.argmax(min_distances)
        # get minimum distance of unselected samples to that sample only
        min_distances = np.minimum(min_distances, ks_distance[:, already_selected[i]])

    return already_selected
