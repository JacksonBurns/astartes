"""
The Optimizable K-Dissimilarity Selection (OptiSim) algorithm, as originally
described by Clark (https://pubs.acs.org/doi/full/10.1021/ci970282v), adapted
to work for arbitrary distance metrics.

The original algorithm:
1. Initialization
 - Take a featurized dataset and select an arbitrary starting data point for
   the selection set.
 - Treat the remaining data as 'candidates'.
 - Create an empty 'recycling bin'.
 - Create an empty subsample set.
 - Create an empty selection set.

2. Remove a random point from the candidates.
 - if it has a similarity greater than a given cutoff to any of the members of the selection set,
 recycle it (or conversely, if it is within a cutoff distance)
 - otherwise, add to subsample set

3. Repeat 2 until one of two conditions is met:
 a. The subsample reaches the pre-determined maximum size K or
 b. The candidates are exhausted.

4. If Step 3 resulted in condition b, move all data from recycling bin and go
to Step 2.

5. If subsample is empty, quit (all remaining candidates are similar, the
most dissimilar data points have already been identified)

6. Pick the most dissimilar (relative to data points already in selection set)
point in the subsample and add it to the selection set.

7. Move the remaining points in the subsample to the recycling bin.

8. If size(selection set) is sufficient, quit. Otherwise, go to Step 2.

As suggested in the original paper, the members of the selection set are then
used as cluster centers, and we assign every element in the dataset to belong
to the cluster containing the selection set member to which it is the most
similar. To implement this step, use scipy.spatial.distance.cdist.

This algorithm seems like it might introduce an infinite loop if the subsample
is not filled and all of the remaining candidates are within the cutoff and cannot
be added. Might need a stop condition here? Unless the empyting of the recycling bin
will somehow fix this. Also possible that one would never have a partially filled
subsample after looking at the full dataset since it is more probable that ALL the
points would be rejected and the subsample would be empty.

Likely just check for no more points being possible to fit into the subsample, and
exit if that is the case.

"""
from math import floor

import numpy as np
from scipy.spatial.distance import pdist

from astartes.samplers import AbstractSampler


class OptiSim(AbstractSampler):
    def __init__(self, *args):
        super().__init__(*args)

    def _sample(self):
        """Implementes the OptiSim sampler"""
        self._init_random(self.get_config("random_state", 42))
        n_samples = len(self.X)

        # pull out the user configs, named as they are in the original algorithm
        # the max number of clusters to find
        M = self.get_config("n_clusters", 10)
        # use up to 5% of the data per iteration in sampler
        K = self.get_config("max_subsample_size", 1 + floor(n_samples * 0.05))
        # up to 10% distance is good enough for default
        c = self.get_config("distance_cutoff", 0.10)

        # all of the sets described in the paper will operate with indices
        selection_set = set()
        subsample_set = set()
        recycling_bin = set()
        candidate_set = set(range(n_samples))

        # pick an arbitrary first member of the selection set
        initial_selection = self.rchoose(candidate_set)
        self.move_item(initial_selection, candidate_set, selection_set)

        # pick an arbitrary starting point, add to subsample, remove from candidates
        start_sample = self.rchoose(candidate_set)
        self.move_item(start_sample, candidate_set, subsample_set)

        # Step 5 in the process is rare and does not fit well with the below implementation
        # since it would require nested while loops. Thus, we will introduce an break counter
        # and use it as a backup (proboably good policy anyway).
        emergency_break = 0
        # worst case, we try putting each sample into each cluster once
        _it_limit = M * n_samples

        # continue as long as selection set is not full (and avoid infinite loop)
        while len(selection_set) < M and emergency_break < _it_limit:
            if len(subsample_set) == K:  # 3a
                # pick subsample member most dissimilar to the selection set
                # by adding the distance of each subsample member from
                # the members of the selection set
                distance_scores = {}
                for sample in subsample_set:
                    distance_scores[sample] = 0.0
                    for selection_sample in selection_set:
                        distance_scores[sample] += self.get_dist(
                            sample, selection_sample
                        )
                furthest_sample = max(distance_scores, key=distance_scores.get)
                self.move_item(furthest_sample, subsample_set, selection_set)
                # move other subsampled items to recycling, clear subsample
                recycling_bin.update(subsample_set)
                subsample_set.clear()
            elif not len(candidate_set):  # 3b
                # move data from recycling to candidate list
                candidate_set.update(recycling_bin)
                recycling_bin.clear()
            else:  # subsample not full, still more candidates, attempt to add
                candidate_sample = self.rchoose(candidate_set)
                if any(  # it is different from all the
                    [self.get_dist(candidate_sample, i) < c for i in selection_set]
                ):
                    self.move_item(candidate_sample, candidate_set, recycling_bin)
                else:
                    self.move_item(candidate_sample, candidate_set, subsample_set)

            emergency_break += 1

        # now we assign the clusters based on which samples are closest to which members
        # of the selection set
        cluster_labels = dict(zip(selection_set, range(len(selection_set))))
        labels = np.full(n_samples, fill_value=-1, dtype=int)
        for sample in range(n_samples):
            if sample in cluster_labels.keys():  # cluster center
                labels[sample] = cluster_labels[sample]
            else:  # find which cluster center is closest
                distances = {
                    i: self.get_dist(sample, i) for i in cluster_labels.values()
                }
                labels[sample] = min(distances, key=distances.get)

        self._samples_clusters = labels

    def _init_random(self, random_state):
        """This uses a lot of random numbers, so make them convenient to reach."""
        self._rng = np.random.default_rng(seed=random_state)
        return

    def rchoose(self, set):
        """Choose a random element from a set with self._rng"""
        return self._rng.choice(list(set))

    def get_dist(self, i, j):
        """Calculates pdist and returns distance between two samples"""
        if not hasattr(self, "_pdist"):
            # calculate the distance matrix, normalize it
            dist_array = pdist(self.X, metric=self.get_config("metric", "euclidean"))
            self._pdist = np.divide(
                dist_array - np.amin(dist_array),
                np.amax(dist_array) - np.amin(dist_array),
            )
        return self._pdist[len(self.X) * i + j - ((i + 2) * (i + 1)) // 2]

    def move_item(self, item, source_set, destintation_set):
        """Moves item from source_set to destination_set"""
        destintation_set.add(item)
        source_set.remove(item)
        return
