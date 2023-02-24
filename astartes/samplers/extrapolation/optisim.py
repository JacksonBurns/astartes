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
 - if it has a similarity greater than a given cutoff to the members of the selection set,
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

from astartes.samplers import AbstractSampler


class OptiSim(AbstractSampler):
    pass
