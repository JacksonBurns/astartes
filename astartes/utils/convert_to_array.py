import warnings

import numpy as np

from astartes.utils.exceptions import UncastableInputError
from astartes.utils.warnings import ConversionWarning


def convert_to_array(obj: object, name: str):
    """Attempt to convert obj named name to a numpy array, with appropriate warnings and exceptions.

    Args:
        obj (object): The item to attempt to convert.
        name (str): Human-readable name for printing.
    """
    warnings.warn(
        "Attempting to cast {:s} to a numpy array, which may result in unexpected behavior "
        "(remove this warning by passing numpy arrays directly to astartes).".format(
            name,
        ),
        ConversionWarning,
    )
    try:
        new_array = np.asarray(obj)
    except Exception as e:
        raise UncastableInputError(
            "Unable to cast {:s} to a numpy array using np.asarray."
        ) from e

    return new_array
