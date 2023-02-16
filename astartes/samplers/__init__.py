# abstract base classes
from .abstract_sampler import AbstractSampler

# implementations
from .extrapolation import KMeans, Scaffold, SphereExclusion
from .interpolation import DBSCAN, MTSD, DOptimal, Duplex, KennardStone, OptiSim, Random

IMPLEMENTED_INTERPOLATION_SAMPLERS = (
    "random",
    # "dbscan",
    # "doptimal",
    # "duplex",
    "kennard_stone",
    # "mtsd",
    # "optisim",
    "sphere_exclusion",
)

IMPLEMENTED_EXTRAPOLATION_SAMPLERS = (
    # "scaffold",
    "kmeans",
)

ALL_SAMPLERS = IMPLEMENTED_EXTRAPOLATION_SAMPLERS + IMPLEMENTED_INTERPOLATION_SAMPLERS
