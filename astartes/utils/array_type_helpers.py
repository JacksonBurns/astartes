import numbers
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd

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
        if not hasattr(item, "__add__"):  # only allow items which we can do math on
            raise InvalidConfigurationError(
                "After casting, input object {:s} contained unsupported type {:s}".format(
                    name,
                    str(type(item)),
                )
            )

    return new_array


def panda_handla(X, y, labels):
    """Helper function to deal with supporting Pandas data types in astartes

    Args:
        X (Dataframe): Features with column names
        y (Series): Targets
        labels (Series): Labels for data

    Returns:
        dict: Empty if no pandas types, metadata-filled otherwise
    """
    output_is_pandas = {}
    # X can be a dataframe, in which case save the column names
    if type(X) is pd.DataFrame:
        output_is_pandas["X"] = {"columns": X.columns, "index": X.index}

    # y can be a series, indicate with True
    if type(y) is pd.Series:  # this implicitly avoids None types
        output_is_pandas["y"] = {"index": y.index}

    # labels can also be a series, indicate with True
    if type(labels) is pd.Series:
        output_is_pandas["labels"] = {"index": labels.index}

    return output_is_pandas


def return_helper(
    sampler_instance,
    train_idxs,
    val_idxs,
    test_idxs,
    return_indices,
    output_is_pandas,
):
    """Convenience function to return the requested arrays appropriately.

    Args:
        sampler_instance (sampler): The fit sampler instance.
        test_size (float): Fraction of data to use in test.
        val_size (float): Fraction of data to use in val.
        train_size (float): Fraction of data to use in train.
        return_indices (bool): Return indices after the value arrays.
        output_is_pandas (dict): metadata about casting to pandas.

    Returns:
        np.array: Either many arrays or indices in arrays.

    Notes:
        This function copies and pastes a lot of code when it could instead
        use some loop over (X, y, labels, sampler_instance.get_clusters())
        but such an implementation is more error prone. This is long and
        not the prettiest, but it is definitely doing what we want.
    """
    # pack all the output into a list and then unpack it in the return statement
    out = []

    # Feature array
    X_train = sampler_instance.X[train_idxs]
    # check if it needs to be converted back to a Dataframe
    dataframe_metadata = output_is_pandas.get("X", False)
    if dataframe_metadata:
        X_train = pd.DataFrame(
            X_train,
            columns=dataframe_metadata["columns"],
            index=dataframe_metadata["index"][train_idxs],
        )
    out.append(X_train)
    # repeat the above process for validation and test
    if len(val_idxs):
        X_val = sampler_instance.X[val_idxs]
        if dataframe_metadata:
            X_val = pd.DataFrame(
                X_val,
                columns=dataframe_metadata["columns"],
                index=dataframe_metadata["index"][val_idxs],
            )
        out.append(X_val)
    X_test = sampler_instance.X[test_idxs]
    if dataframe_metadata:
        X_test = pd.DataFrame(
            X_test,
            columns=dataframe_metadata["columns"],
            index=dataframe_metadata["index"][test_idxs],
        )
    out.append(X_test)

    # repeat above with y, if it exists
    if sampler_instance.y is not None:
        y_train = sampler_instance.y[train_idxs]
        # if user passed y as a series, recast it
        series_metadata = output_is_pandas.get("y", False)
        if series_metadata:
            y_train = pd.Series(
                y_train,
                index=series_metadata["index"][train_idxs],
            )
        out.append(y_train)
        if len(val_idxs):
            y_val = sampler_instance.y[val_idxs]
            if series_metadata:
                y_val = pd.Series(
                    y_val,
                    index=series_metadata["index"][val_idxs],
                )
            out.append(y_val)
        y_test = sampler_instance.y[test_idxs]
        if series_metadata:
            y_test = pd.Series(
                y_test,
                index=series_metadata["index"][test_idxs],
            )
        out.append(y_test)

    # repeat with labels, if they exist
    if sampler_instance.labels is not None:
        labels_train = sampler_instance.labels[train_idxs]
        # if user passed labels as a series, recast it
        series_metadata = output_is_pandas.get("labels", False)
        if series_metadata:
            labels_train = pd.Series(
                labels_train,
                index=series_metadata["index"][train_idxs],
            )
        out.append(labels_train)
        if len(val_idxs):
            labels_val = sampler_instance.labels[val_idxs]
            if series_metadata:
                labels_val = pd.Series(
                    labels_val,
                    index=series_metadata["index"][val_idxs],
                )
            out.append(labels_val)
        labels_test = sampler_instance.labels[test_idxs]
        if series_metadata:
            labels_test = pd.Series(
                labels_test,
                index=series_metadata["index"][test_idxs],
            )
        out.append(labels_test)

    # repeat above with the clusters, if a clustering approach was used
    if len(sampler_instance.get_clusters()):  # true when the list has been filled
        clusters_train = sampler_instance.get_clusters()[train_idxs]
        out.append(clusters_train)
        if len(val_idxs):
            clusters_val = sampler_instance.get_clusters()[val_idxs]
            out.append(clusters_val)
        clusters_test = sampler_instance.get_clusters()[test_idxs]
        out.append(clusters_test)

    # append indices, if requested
    if return_indices:
        out.append(train_idxs)
        if val_idxs.any():
            out.append(val_idxs)
        out.append(test_idxs)

    return (*out,)
