# convenience import to enable 'from astartes import train_test_split
from .astartes import train_test_split

# DO NOT do this:
# from .molecules import train_test_split_molecules
# on installations without the optional molecules install, this would cause
# an exception

from .astartes import IMPLEMENTED_SAMPLERS


__version__ = "0.0.0"
