# abstract base classes
from .abstract_sampler import AbstractSampler

# implementations
from .interpolation import (
    Random,
    DBSCAN,
    DOptimal,
    Duplex,
    KennardStone,
    MTSD,
    OptiSim,
)
from .extrapolation import (
    Scaffold,
    KMeans,
    SphereExclusion,
)


IMPLEMENTED_INTERPOLATION_SAMPLERS = (
    "random",
    # "dbscan",
    # "doptimal",
    # "duplex",
    # "kennard_stone",
    # "mtsd",
    # "optisim",
    "sphere_exclusion",
)

IMPLEMENTED_EXTRAPOLATION_SAMPLERS = (
    # "scaffold",
    "kmeans",
)

ALL_SAMPLERS = IMPLEMENTED_EXTRAPOLATION_SAMPLERS + IMPLEMENTED_INTERPOLATION_SAMPLERS
