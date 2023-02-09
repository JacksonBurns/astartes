# abstract base classes
from .abstract_unsupervised_sampler import AbstractUnsupervisedSampler
from .abstract_supervised_sampler import AbstractSupervisedSampler

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
