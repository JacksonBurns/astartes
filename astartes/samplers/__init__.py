# abstract base classes
from .abstract_sampler import AbstractSampler

# implementations
from .extrapolation import DBSCAN, KMeans, OptiSim, Scaffold, SphereExclusion, TimeBased, TargetProperty
from .interpolation import SPXY, KennardStone, Random

IMPLEMENTED_INTERPOLATION_SAMPLERS = (
    "random",
    "kennard_stone",
    "spxy",
)

IMPLEMENTED_EXTRAPOLATION_SAMPLERS = (
    "dbscan",
    "scaffold",
    "kmeans",
    "molecular_weight",
    "optisim",
    "sphere_exclusion",
    "time_based",
    "target_property",
)

ALL_SAMPLERS = IMPLEMENTED_EXTRAPOLATION_SAMPLERS + IMPLEMENTED_INTERPOLATION_SAMPLERS
