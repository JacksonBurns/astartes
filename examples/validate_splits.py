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

split_names = ["train", "val", "test"]
failures = []
for reference, new in tests_dict.items():
    with open(reference, "rb") as f:
        reference_splits = pkl.load(f)
    with open(new, "rb") as f:
        new_splits = pkl.load(f)
    for split_idx in range(5):
        for set_idx in range(3):
            try:
                np.testing.assert_array_equal(
                    reference_splits[split_idx][set_idx],
                    new_splits[split_idx][set_idx],
                    "Failed to reproduce {:s} split {:d} on {:s} set.".format(
                        new,
                        split_idx,
                        split_names[set_idx],
                    ),
                )
            except AssertionError as ae:
                failures.append(str(ae))
if failures:
    raise RuntimeError(
        "The following splits were not reproduced successfully: " + repr(failures)
    )
