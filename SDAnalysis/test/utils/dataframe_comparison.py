import pandas as pd
import numpy as np


def dataframe_differences(df1: pd.DataFrame, df2: pd.DataFrame, both_nan_equal: bool = False):
    """Given two dataframes with the same shape and columns, return a boolean dataframe of the same shape filled with True where the entries are equal,
    and False where they are not equal. If a position in both dataframes is np.NaN, the comparison for that cell is evaluated as "both_nan_equal"

    Parameters
    ----------
    df1 : pd.DataFrame
        first dataframe to compare
    df2 : pd.DataFrame
        second dataframe to compare
    both_nan_equal : bool, optional
        the value to fill cells of resulting boolean dataframe where both df1 and df2 contains np.NaN, by default False

    Returns
    -------
    pd.DataFrame
        a boolean dataframe of the same shape as the input dataframes, filled with True where the entries are equal and not np.NaN, both_nan_equal where equal and np.NaN, and False where they are not equal
    """

    if not (isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame)):
        raise TypeError("Both inputs must be pandas DataFrames.")
    if not df1.shape == df2.shape:
        raise ValueError("Dataframes must have the same shape.")
    if not (df1.columns == df2.columns).all():
        raise ValueError("Dataframes must have the same columns.")

    # Tolerance for float comparison
    tolerance = 1e-3

    # Initialize an empty DataFrame for the mask with the same shape
    mask = pd.DataFrame(True, index=df1.index, columns=df1.columns)

    # Loop through each column and compare values based on type
    for col in df1.columns:
        # Check for approximate equality for floats
        if np.issubdtype(df1[col].dtype, np.floating):
            mask[col] = np.isclose(df1[col], df2[col], atol=tolerance)
        else:  # Check for exact equality for other types
            mask[col] = df1[col] == df2[col]
        mask[col][df1[col].isna() & df2[col].isna()] = both_nan_equal
    return mask


def dataframes_equal(df1, df2, both_nan_equal: bool = False):
    """Compare two dataframes for equality. If a position in both dataframes is np.NaN, the comparison for that cell is evaluated as True.
    Parameters
    ----------
    df1 : pandas.DataFrame
        One dataframe to compare
    df2 : pandas.DataFrame
        Another dataframe to compare
    both_nan_equal : bool, optional
        Whether to consider np.NaN in both dataframes as equal, by default False
    Returns
    -------
    bool
        True if the dataframes are equal, False otherwise
    """
    if not (isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame)):
        raise TypeError("Both inputs must be pandas DataFrames.")
    if not df1.shape == df2.shape:
        return False
    if not (df1.columns == df2.columns).all():
        return False
    if not (df1.index == df2.index).all():
        return False
    comparison = dataframe_differences(df1, df2, both_nan_equal=both_nan_equal)
    return comparison.all().all()  # collapse over rows and then over columns
