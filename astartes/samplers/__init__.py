# abstract base classes
from .abstract_sampler import AbstractSampler

# implementations
from .extrapolation import DBSCAN, KMeans, Scaffold, SphereExclusion, OptiSim
from .interpolation import MTSD, DOptimal, Duplex, KennardStone, Random

IMPLEMENTED_INTERPOLATION_SAMPLERS = (
    "random",
    # "doptimal",
    # "duplex",
    "kennard_stone",
    # "mtsd",
    "sphere_exclusion",
)

IMPLEMENTED_EXTRAPOLATION_SAMPLERS = (
    "dbscan",
    # "scaffold",
    "kmeans",
    "optisim",
)

ALL_SAMPLERS = IMPLEMENTED_EXTRAPOLATION_SAMPLERS + IMPLEMENTED_INTERPOLATION_SAMPLERS
