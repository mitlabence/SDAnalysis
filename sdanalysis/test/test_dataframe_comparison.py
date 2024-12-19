"""
test_dataframe_comparison.py - test the dataframe comparison utility functions
"""

import pandas as pd
import utils.dataframe_comparison as dc
import numpy as np


class TestDataframeEquals:
    """
    Test the dataframes_equal function in the dataframe_comparison module
    """
    def test_int_equal(self):
        """
        Test that two dataframes with integer values are equal
        """
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert dc.dataframes_equal(df1, df2)
        assert dc.dataframes_equal(df2, df1)

    def test_int_not_equal(self):
        """
        Test that two dataframes with integer values (with difference in one position) are not equal
        """
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})
        assert not dc.dataframes_equal(df1, df2)
        assert not dc.dataframes_equal(df2, df1)

    def test_float_equal(self):
        """
        Test that two dataframes with float values are equal
        """
        df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        assert dc.dataframes_equal(df1, df2)
        assert dc.dataframes_equal(df2, df1)

    def test_float_not_equal(self):
        """
        Test that two dataframes with float values (with difference in one position) are not equal
        """
        df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 7.0]})
        assert not dc.dataframes_equal(df1, df2)
        assert not dc.dataframes_equal(df2, df1)

    def test_float_with_nan_equal_both_true(self):
        """
        Test that two dataframes with same float values (with np.NaN in common locations) are equal
        when both_nan_equal is set to True
        """
        df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [4.0, 5.0, 6.0, np.nan]})
        df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [4.0, 5.0, 6.0, np.nan]})
        assert dc.dataframes_equal(df1, df2, both_nan_equal=True)
        assert dc.dataframes_equal(df2, df1, both_nan_equal=True)

    def test_float_with_nan_equal_both_false(self):
        """
        Test that two dataframes with same float values (with np.NaN in common locations) are not equal
        when both_nan_equal is set to False
        """
        df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [4.0, 5.0, 6.0, np.nan]})
        df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [4.0, 5.0, 6.0, np.nan]})
        assert not dc.dataframes_equal(df1, df2, both_nan_equal=False)
        assert not dc.dataframes_equal(df2, df1, both_nan_equal=False)

    def test_mixed_type_equals(self):
        """
        Test that two dataframes with mixed types are equal
        """
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        assert df1["a"].dtype == np.int64
        assert df1["b"].dtype == np.float64
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert df2["a"].dtype == np.int64
        assert df2["b"].dtype == np.int64
        assert dc.dataframes_equal(df1, df2)
        assert dc.dataframes_equal(df2, df1)

    def test_float_tolerance(self):
        """
        Test that two dataframes with float values are equal within a given tolerance
        """
        df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.001]})
        assert dc.dataframes_equal(df1, df2, tolerance=0.01)
        assert dc.dataframes_equal(df2, df1, tolerance=0.01)
        assert not dc.dataframes_equal(df1, df2, tolerance=0.0009)
        assert not dc.dataframes_equal(df2, df1, tolerance=0.0009)


class TestSeriesEquals:
    """
    Test the series_equal function in the dataframe_comparison module
    """
    def test_int_equal(self):
        """
        Test that two series with integer values are equal
        """
        s1 = pd.Series([1, 2, 3])
        s2 = pd.Series([1, 2, 3])
        assert dc.series_equal(s1, s2)
        assert dc.series_equal(s2, s1)

    def test_int_not_equal(self):
        """
        Test that two series with integer values (with difference in one position) are not equal
        """
        s1 = pd.Series([1, 2, 3])
        s2 = pd.Series([1, 2, 4])
        assert not dc.series_equal(s1, s2)
        assert not dc.series_equal(s2, s1)

    def test_float_equal(self):
        """
        Test that two series with float values are equal
        """
        s1 = pd.Series([1.0, 2.0, 3.0])
        s2 = pd.Series([1.0, 2.0, 3.0])
        assert dc.series_equal(s1, s2)
        assert dc.series_equal(s2, s1)

    def test_float_not_equal(self):
        """
        Test that two series with float values (with difference in one position) are not equal
        """
        s1 = pd.Series([1.0, 2.0, 3.0])
        s2 = pd.Series([1.0, 2.0, 4.0])
        assert not dc.series_equal(s1, s2)
        assert not dc.series_equal(s2, s1)

    def test_float_with_nan_equal_both_true(self):
        """
        Test that two series with same float values (with np.NaN in common locations) are equal
        when both_nan_equal is set to True
        """
        s1 = pd.Series([1.0, 2.0, 3.0, np.nan])
        s2 = pd.Series([1.0, 2.0, 3.0, np.nan])
        assert dc.series_equal(s1, s2, both_nan_equal=True)
        assert dc.series_equal(s2, s1, both_nan_equal=True)

    def test_float_with_nan_equal_both_false(self):
        """
        Test that two series with same float values (with np.NaN in common locations) are not equal
        when both_nan_equal is set to False
        """
        s1 = pd.Series([1.0, 2.0, 3.0, np.nan])
        s2 = pd.Series([1.0, 2.0, 3.0, np.nan])
        assert not dc.series_equal(s1, s2, both_nan_equal=False)
        assert not dc.series_equal(s2, s1, both_nan_equal=False)

    def test_float_with_nan_equal_one_true_one_false(self):
        """
        Test that two series with same float values (with np.NaN in different locations) are not equal
        when one both_nan_equal is set to True and the other is set to False
        """
        s1 = pd.Series([1.0, 2.0, 3.0, np.nan])
        s2 = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert not dc.series_equal(s1, s2, both_nan_equal=False)
        assert not dc.series_equal(s1, s2, both_nan_equal=True)

        s1 = pd.Series([1.0, 2.0, 3.0, 4.0])
        s2 = pd.Series([1.0, 2.0, 3.0, np.nan])
        assert not dc.series_equal(s1, s2, both_nan_equal=False)
        assert not dc.series_equal(s1, s2, both_nan_equal=True)

        s1 = pd.Series([1.0, 2.0, 3.0, np.nan])
        s2 = pd.Series([1.0, 2.0, np.nan, 3.0])
        assert not dc.series_equal(s1, s2, both_nan_equal=False)
        assert not dc.series_equal(s1, s2, both_nan_equal=True)
