"""
test_dataframe_comparison.py - test the dataframe comparison utility functions
"""

import pandas as pd
import utils.dataframe_comparison as dc
import numpy as np

# TODO: add more tests


def test_int_equal():
    """
    Test that two dataframes with integer values are equal
    """
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert dc.dataframes_equal(df1, df2)


def test_int_not_equal():
    """
    Test that two dataframes with integer values (with difference in one position) are not equal
    """
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})
    assert not dc.dataframes_equal(df1, df2)


def test_float_equal():
    """
    Test that two dataframes with float values are equal
    """
    df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    assert dc.dataframes_equal(df1, df2)


def test_float_not_equal():
    """
    Test that two dataframes with float values (with difference in one position) are not equal
    """
    df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 7.0]})
    assert not dc.dataframes_equal(df1, df2)


def test_float_with_nan_equal_both_true():
    """
    Test that two dataframes with same float values (with np.NaN in common locations) are equal
    when both_nan_equal is set to True
    """
    df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [4.0, 5.0, 6.0, np.nan]})
    df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [4.0, 5.0, 6.0, np.nan]})
    assert dc.dataframes_equal(df1, df2, both_nan_equal=True)


def test_float_with_nan_equal_both_false():
    """
    Test that two dataframes with same float values (with np.NaN in common locations) are not equal
    when both_nan_equal is set to False
    """
    df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [4.0, 5.0, 6.0, np.nan]})
    df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [4.0, 5.0, 6.0, np.nan]})
    assert not dc.dataframes_equal(df1, df2, both_nan_equal=False)
