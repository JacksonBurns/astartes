"""
The Sphere Exclusion clustering algorithm.

This re-implementation draws from this blog post on the RDKit blog,
though abstracted to work for arbitrary feature vectors:
http://rdkit.blogspot.com/2020/11/sphere-exclusion-clustering-with-rdkit.html
As well as this paper:
https://www.daylight.com/cheminformatics/whitepapers/ClusteringWhitePaper.pdf

But instead of using tanimoto similarity, which has a domain between zero and
one, it uses euclidian distance to enable processing arbitrary valued
vectors.
"""
from math import floor

import numpy as np
from scipy.spatial.distance import pdist, squareform

from astartes.samplers import AbstractSampler


class SphereExclusion(AbstractSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _sample(self):
        """Cluster X according to a Sphere Exclusion-like algorithm with arbitrary distance metrics."""
        # euclidian, cosine, or city block from get_configs
        # calculate pdist
        dist_array = pdist(self.X, metric=self.get_config("metric", "euclidean"))
        dist_array = squareform(
            np.divide(  # normalize it st highest similarity is 1
                dist_array - np.amin(dist_array),
                np.amax(dist_array) - np.amin(dist_array),
            )
        )

        # at most, every row will be in its own cluster, so create a list of indices
        # and just iterate through it, skipping those that are already clustered
        rng = np.random.default_rng(self.get_config("random_state", None))
        idxs = np.arange(len(self.X))
        rng.shuffle(idxs)

        # use a default similarity or get_config
        distance_cutoff = self.get_config("distance_cutoff", 0.25)

        # build first cluster (add those that are clustered with it to a set as you go)
        # and continue until every row is sorted
        already_assigned = set()
        # output labels
        labels = np.full(len(self.X), fill_value=-1, dtype=int)
        # label counter
        cluster_idx = 0
        for sample_idx in idxs:
            if sample_idx in already_assigned:  # skip rows that are already assigned
                continue
            # get the row of the distance matrix for that sample
            row = dist_array[sample_idx]
            # find indices where it is close (given distance cutoff) including itself
            candidate_indices = set(np.flatnonzero(row < distance_cutoff))
            # check which of these have not already been assigned
            unassigned_indices = candidate_indices.difference(already_assigned)
            # add the labels to these
            for i in unassigned_indices:
                labels[i] = cluster_idx
            # increment the cluster index
            cluster_idx += 1
            # add these used indices to the tracker
            already_assigned.update(unassigned_indices)

        self._samples_clusters = labels
