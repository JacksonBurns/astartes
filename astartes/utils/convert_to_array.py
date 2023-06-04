import warnings
from datetime import date, datetime

import numpy as np

from astartes.utils.exceptions import InvalidConfigurationError, UncastableInputError
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
    # ensure that all the values in the array are floats or date/datetime objects
    for item in new_array.ravel():
        if type(item) not in (float, np.float_, int, np.int_, date, datetime):
            raise InvalidConfigurationError(
                "After casting, input object {:s} contained unsupported type {:s}".format(
                    name,
                    str(type(item)),
                )
            )

    return new_array
