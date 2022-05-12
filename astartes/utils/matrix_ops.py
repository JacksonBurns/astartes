import math


def square_to_condensed(row, col, n_samples):
    """
    Get the condensed_id corresponding to the (row, col) tuple
    in the squareform matrix.
    CondensedMatrix[condensed_id] == SquareformMatrix[row, col]
    Args:
        row (int): Row id in the squareform matrix.
        col (int): Column id in the squareform matrix.
        n_samples (int): Number of samples.
    Returns:
        (int): conensed_id
    """
    assert row != col, "no diagonal elements in condensed matrix"
    if row < col:
        row, col = col, row
    return n_samples*col - col*(col+1)//2 + row - 1 - col


def calc_row_idx(condensed_id, n_samples):
    """
    Get the row id in the square form matrix corresponding to
    the condensed_id in the condensed matrix.
    Args:
        condensed_id (int): Id in the condensed matrix.
        n_samples (int): Number of samples.
        The formula is derived from the quadratic form.
    Returns:
        (int): Row index.
    """
    return int(math.ceil((1/2.) * (- (-8*condensed_id
                                      + 4 * n_samples**2
                                      - 4 * n_samples
                                      - 7)**0.5 + 2 * n_samples - 1) - 1))


def total_n_elems_upto_row(row, n_samples):
    """
    Get the total number of elements up to the current row of the lower
    triangular matrix (diaganols are excluded).
    Args:
        row (int): Row id in the traingular matrix.
        n_samples (int): Number of samples.
    Returns:
        (int): Number of elements in the row.
    """
    return row * (n_samples - 1 - row) + (row*(row + 1))//2


def calc_col_idx(condensed_id, row, n_samples):
    """
    Get the column id in the square form matrix corresponding to
    the condensed_id in the condensed matrix.
    Args:
        condensed_id (int): Id in the condensed matrix.
        row (int): Row id in the squareform matrix.
        n_samples (int): Number of samples.
    Returns:
        (int): Column index.
    """
    return int(n_samples - total_n_elems_upto_row(row + 1, n_samples)
               + condensed_id)


def condensed_to_square(condensed_id, n_samples):
    """
    Get the (row, col) tuple in the squareform matrix corresponding to the
    condensed_id in the condensed matrix.
    CondensedMatrix[condensed_id] == SquareformMatrix[row, col]
    Args:
        condensed_id (int): Id in the condensed matrix.
        n_samples (int): Number of samples.
    Returns:
        tuple: (row_id, col_id) in the squareform matrix.
    """
    row = calc_row_idx(condensed_id, n_samples)
    col = calc_col_idx(condensed_id, row, n_samples)
    return row, col
