from difflib import get_close_matches

from astartes.samplers import (
    ALL_SAMPLERS,
    DBSCAN,
    SPXY,
    KennardStone,
    KMeans,
    OptiSim,
    Random,
    Scaffold,
    SphereExclusion,
)
from astartes.utils.exceptions import SamplerNotImplementedError


class SamplerFactory:
    def __init__(self, sampler):
        """Initialize SamplerFactory and copy a lowercased 'sampler' into an attribute.

        Args:
            sampler (string): The desired sampler.
        """
        self.sampler = sampler.lower()

    def get_sampler(self, X, y, labels, hopts):
        """Instantiate (which also performs fitting) and return the sampler.

        Args:
            X (np.array): Feature array.
            y (np.array): Target array.
            labels (np.array): Label array.
            hopts (dict): Hyperparameters for the sampler.

        Raises:
            SamplerNotImplementedError: Raised when an non-existent or not yet implemented sampler is requested.

        Returns:
            astartes.sampler: The fit sampler instance.
        """
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
        elif self.sampler == "optisim":
            sampler_class = OptiSim
        elif self.sampler == "spxy":
            sampler_class = SPXY
        elif self.sampler == "scaffold":
            sampler_class = Scaffold
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
            raise SamplerNotImplementedError(
                "Sampler {:s} has not been implemented.".format(self.sampler) + addendum
            )
        return sampler_class(X, y, labels, hopts)
