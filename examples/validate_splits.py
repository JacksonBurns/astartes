"""
validate_splits.py - used in GitHub actions to verify that splits produced by
subsequent releases of astartes still reproduce the data splits published
in the original paper using astartes v1.0.0

See .github/workflows/reproduce_paper.yml to see where this is used.
In summary, we do this:
 - on a schedule, on pull requests, or on push run the reproduce_paper.yml action
 - make a backup copy of all the reference splits in examples/*_splits
 - run the *_make_splits notebooks to make the splits using the current astartes
 version (including the latest pushes, if on a PR)
 - run this script to compare these
"""
import pickle as pkl

import numpy as np

tests_dict = {
    "REFERENCE_RDB7_splits_kmeans.pkl": "RDB7_splits/RDB7_splits_kmeans.pkl",
    "REFERENCE_RDB7_splits_random.pkl": "RDB7_splits/RDB7_splits_random.pkl",
    "REFERENCE_RDB7_splits_scaffold.pkl": "RDB7_splits/RDB7_splits_scaffold.pkl",
    "REFERENCE_QM9_splits_kmeans.pkl": "QM9_splits/QM9_splits_kmeans.pkl",
    "REFERENCE_QM9_splits_random.pkl": "QM9_splits/QM9_splits_random.pkl",
    "REFERENCE_QM9_splits_scaffold.pkl": "QM9_splits/QM9_splits_scaffold.pkl",
}

for reference, new in tests_dict.items():
    with open(reference, "rb") as f:
        reference_splits = pkl.load(f)
    with open(new, "rb") as f:
        new_splits = pkl.load(f)

    np.testing.assert_array_equal(
        reference_splits,
        new_splits,
        "Failed to reproduce {:s}.".format(new),
    )
