from difflib import get_close_matches

from astartes.samplers import (
    ALL_SAMPLERS,
    DBSCAN,
    KennardStone,
    KMeans,
    Random,
    SphereExclusion,
)
from astartes.utils.exceptions import NotImplementedError


class SamplerFactory:
    def __init__(self, sampler):
        self.sampler = sampler.lower()

    def get_sampler(self, X, y, labels, hopts):
        if self.sampler == "random":
            sampler_class = Random
        elif self.sampler == "kennard_stone":
            sampler_class = KennardStone
        elif self.sampler == "dbscan":
            sampler_class = DBSCAN
        elif self.sampler == "kmeans":
            sampler_class = KMeans
        elif self.sampler == "sphere_exclusion":
            sampler_class = SphereExclusion
        else:
            possiblity = get_close_matches(
                self.sampler,
                ALL_SAMPLERS,
                n=1,
            )
            addendum = (
                " Did you mean '{:s}'?".format(possiblity[0])
                if possiblity
                else " Try help(train_test_split)."
            )
            raise NotImplementedError(
                "Sampler {:s} has not been implemented.".format(self.sampler) + addendum
            )
        return sampler_class(X, y, labels, hopts)
