# abstract base classes
from .abstract_sampler import AbstractSampler

# implementations
from .extrapolation import DBSCAN, KMeans, OptiSim, Scaffold, SphereExclusion
from .interpolation import MTSD, SPXY, DOptimal, Duplex, KennardStone, Random

IMPLEMENTED_INTERPOLATION_SAMPLERS = (
    "random",
    # "doptimal",
    # "duplex",
    "kennard_stone",
    # "mtsd",
    "spxy",
)

IMPLEMENTED_EXTRAPOLATION_SAMPLERS = (
    "dbscan",
    "scaffold",
    "kmeans",
    "optisim",
    "sphere_exclusion",
)

ALL_SAMPLERS = IMPLEMENTED_EXTRAPOLATION_SAMPLERS + IMPLEMENTED_INTERPOLATION_SAMPLERS
