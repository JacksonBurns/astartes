# convenience import to enable 'from astartes import train_test_split'
from .main import train_test_split, train_val_test_split

__version__ = "1.3.3"

# DO NOT do this:
# from .molecules import train_test_split_molecules
# on installations without the optional molecules install, this would cause
# an exception
