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
    # "REFERENCE_RDB7_splits_kmeans.pkl": "RDB7_splits/RDB7_splits_kmeans.pkl",
    "REFERENCE_RDB7_splits_random.pkl": "RDB7_splits/RDB7_splits_random.pkl",
    "REFERENCE_RDB7_splits_scaffold.pkl": "RDB7_splits/RDB7_splits_scaffold.pkl",
    # "REFERENCE_QM9_splits_kmeans.pkl": "QM9_splits/QM9_splits_kmeans.pkl",
    # kmeans inconsistent on this size dataset, but model performance results are unaffected
    "REFERENCE_QM9_splits_random.pkl": "QM9_splits/QM9_splits_random.pkl",
    "REFERENCE_QM9_splits_scaffold.pkl": "QM9_splits/QM9_splits_scaffold.pkl",
}

# KMeans assigns labels randomly even with random_seed fixed, which will cause at most
# 2 clusters to be sorted into a different category on subsequent runs
KMEANS_REPRODUCIBILITY_TARGET = 0.9  # 90% the same

split_names = ["train", "val", "test"]
fail = False
for reference, new in tests_dict.items():
    with open(reference, "rb") as f:
        reference_splits = pkl.load(f)
    with open(new, "rb") as f:
        new_splits = pkl.load(f)
    for split_idx in range(5):
        for set_idx in range(3):
            reference_length = len(reference_splits[split_idx][set_idx])
            new_length = len(new_splits[split_idx][set_idx])
            shared_indexes = np.intersect1d(
                reference_splits[split_idx][set_idx], new_splits[split_idx][set_idx]
            )
            largest_split = max(reference_length, new_length)
            shared_percent = len(shared_indexes) / largest_split
            print(
                "\n{:s} split {:d} on {:s} set.".format(
                    new,
                    split_idx,
                    split_names[set_idx],
                )
            )
            print(
                "Dynamic size: {:d} Reference size: {:d}".format(
                    new_length, reference_length
                )
            )
            print(
                "Dynamically generated and reference split shared {:.4f}% of indexes.".format(
                    shared_percent * 100
                )
            )
            are_eq = np.array_equal(
                reference_splits[split_idx][set_idx],
                new_splits[split_idx][set_idx],
            )
            print(
                "Are arrays exactly equal: ",
                are_eq,
            )
            if not are_eq:
                fail = True
if fail:
    raise RuntimeError("Regression testing failed, see above output.")
