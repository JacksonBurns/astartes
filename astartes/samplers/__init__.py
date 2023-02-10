# abstract base classes
from .abstract_sampler import AbstractSampler

# implementations
from .unsupervised import (
    Random,
    DBSCAN,
    DOptimal,
    Duplex,
    KennardStone,
    MTSD,
    OptiSim,
    SphereExclusion,
)
from .supervised import (
    Scaffold,
)


IMPLEMENTED_UNSUPERVISED_SAMPLERS = (
    "random",
    "dbscan",
    "doptimal",
    "duplex",
    "kennard_stone",
    "mtsd",
    "optisim",
    "sphere_exclusion",
)

IMPLEMENTED_SUPERVISED_SAMPLERS = ("scaffold",)
