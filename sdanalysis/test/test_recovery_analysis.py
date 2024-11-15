"""
test_recovery_analysis.py - This file tests the recovery_analysis module.
"""

import os
import sys
import pytest

# import pandas as pd
# from utils.dataframe_comparison import dataframes_equal

try:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root_dir)
finally:
    # from env_reader import read_env
    from recovery_analysis import RecoveryAnalysisData  # , main


class TestRecoveryAnalysisData:
    @pytest.fixture(name="data1", scope="class")
    def fixture_data1(self):
        dict_bl_fluo1 = {"key1": [0, 1, 2]}
        dict_mid_fluo1 = {"key2": [3, 4, 5]}
        dict_post_fluo1 = {"key3": [6, 7, 8]}
        dict_meta1 = {"key4": [9, 10, 11]}
        dict_segment_break_points1 = {"key5": [12, 13, 14]}
        dict_excluded1 = {"key6": [15, 16, 17]}
        return RecoveryAnalysisData(
            dict_bl_fluo1,
            dict_mid_fluo1,
            dict_post_fluo1,
            dict_meta1,
            dict_segment_break_points1,
            dict_excluded1,
        )

    @pytest.fixture(name="data1_copy", scope="class")
    def fixture_data1_copy(self):
        dict_bl_fluo1 = {"key1": [0, 1, 2]}
        dict_mid_fluo1 = {"key2": [3, 4, 5]}
        dict_post_fluo1 = {"key3": [6, 7, 8]}
        dict_meta1 = {"key4": [9, 10, 11]}
        dict_segment_break_points1 = {"key5": [12, 13, 14]}
        dict_excluded1 = {"key6": [15, 16, 17]}
        return RecoveryAnalysisData(
            dict_bl_fluo1,
            dict_mid_fluo1,
            dict_post_fluo1,
            dict_meta1,
            dict_segment_break_points1,
            dict_excluded1,
        )

    @pytest.fixture(name="data2", scope="class")
    def fixture_data2(self):
        dict_bl_fluo2 = {"key1_2": [50, 51, 52]}
        dict_mid_fluo2 = {"key2_2": [53, 54, 55]}
        dict_post_fluo2 = {"key3_2": [56, 57, 58]}
        dict_meta2 = {"key4_2": [69, 60, 61]}
        dict_segment_break_points2 = {"key5_2": [62, 63, 64]}
        dict_excluded2 = {"key6_2": [65, 66, 67]}
        return RecoveryAnalysisData(
            dict_bl_fluo2,
            dict_mid_fluo2,
            dict_post_fluo2,
            dict_meta2,
            dict_segment_break_points2,
            dict_excluded2,
        )

    @pytest.fixture(name="data1_plus_2", scope="class")
    def fixture_data1_plus_2(self):
        dict_bl_fluo12 = {"key1": [0, 1, 2], "key1_2": [50, 51, 52]}
        dict_mid_fluo12 = {"key2": [3, 4, 5], "key2_2": [53, 54, 55]}
        dict_post_fluo12 = {"key3": [6, 7, 8], "key3_2": [56, 57, 58]}
        dict_meta12 = {"key4": [9, 10, 11], "key4_2": [69, 60, 61]}
        dict_segment_break_points12 = {"key5": [12, 13, 14], "key5_2": [62, 63, 64]}
        dict_excluded12 = {"key6": [15, 16, 17], "key6_2": [65, 66, 67]}
        return RecoveryAnalysisData(
            dict_bl_fluo12,
            dict_mid_fluo12,
            dict_post_fluo12,
            dict_meta12,
            dict_segment_break_points12,
            dict_excluded12,
        )

    def test_copy(self):
        """Test the copy method of the RecoveryAnalysisData class.
        It should 1) not be the same object, 2) make an identical copy of the object, and 3) the original should be unchanged
        Args:
            data1 (_type_): _description_
        """
        data = RecoveryAnalysisData(
            {"key1": [0, 1, 2]},
            {"key2": [3, 4, 5]},
            {"key3": [6, 7, 8]},
            {"key4": [9, 10, 11]},
            {"key5": [12, 13, 14]},
            {"key6": [15, 16, 17]},
        )
        data_same = RecoveryAnalysisData(
            {"key1": [0, 1, 2]},
            {"key2": [3, 4, 5]},
            {"key3": [6, 7, 8]},
            {"key4": [9, 10, 11]},
            {"key5": [12, 13, 14]},
            {"key6": [15, 16, 17]},
        )
        assert data is not data_same
        data_copy = data.copy()
        # 1) not be the same object
        assert data is not data_copy
        # 2) make an identical copy of the object
        assert data.dict_bl_fluo == data_copy.dict_bl_fluo
        assert data.dict_mid_fluo == data_copy.dict_mid_fluo
        assert data.dict_post_fluo == data_copy.dict_post_fluo
        assert data.dict_meta == data_copy.dict_meta
        assert data.dict_segment_break_points == data_copy.dict_segment_break_points
        assert data.dict_excluded == data_copy.dict_excluded
        # 3) the original should be unchanged
        assert data.dict_bl_fluo == data_same.dict_bl_fluo
        assert data.dict_mid_fluo == data_same.dict_mid_fluo
        assert data.dict_post_fluo == data_same.dict_post_fluo
        assert data.dict_meta == data_same.dict_meta
        assert data.dict_segment_break_points == data_same.dict_segment_break_points
        assert data.dict_excluded == data_same.dict_excluded

    def test_equals(self, data1, data1_copy):
        """Test equality of two RecoveryAnalysisData objects

        Args:
            data1 (RecoveryAnalysisData): _description_
            data1_copy (RecoveryAnalysisData): _description_
        """
        # none of the two data objects' attributes are the same
        assert data1 == data1_copy
        # for sake of security, test attributes individually
        assert data1.dict_bl_fluo == data1_copy.dict_bl_fluo
        assert data1.dict_mid_fluo == data1_copy.dict_mid_fluo
        assert data1.dict_post_fluo == data1_copy.dict_post_fluo
        assert data1.dict_meta == data1_copy.dict_meta
        assert data1.dict_segment_break_points == data1_copy.dict_segment_break_points
        assert data1.dict_excluded == data1_copy.dict_excluded

    def test_not_equals(self, data1, data2):
        """Test inequality of two RecoveryAnalysisData objects

        Args:
            data1 (RecoveryAnalysisData): _description_
            data2 (RecoveryAnalysisData): _description_
        """
        assert data1 != data2
        # for sake of security, test attributes individually
        assert data1.dict_bl_fluo != data2.dict_bl_fluo
        assert data1.dict_mid_fluo != data2.dict_mid_fluo
        assert data1.dict_post_fluo != data2.dict_post_fluo
        assert data1.dict_meta != data2.dict_meta
        assert data1.dict_segment_break_points != data2.dict_segment_break_points
        assert data1.dict_excluded != data2.dict_excluded

    def test_add(self, data1, data2, data1_plus_2):
        """Test the + operator in two ways: a + b, and a += b.
        Test that the original data is changed or unchanged, depending on usage mode.

        Args:
            data1 (RecoveryAnalysisData): _description_
            data2 (RecoveryAnalysisData): _description_
            data1_plus_2 (RecoveryAnalysisData): _description_
        """
        # test c = a + b, a and b unchanged, c is sum
        data1_c = data1.copy()
        data2_c = data2.copy()
        data_sum = data1 + data2
        assert data_sum == data1_plus_2
        assert data1 == data1_c
        assert data2 == data2_c
        # test a += b, a is sum, b unchanged
        data = data1.copy()
        data += data2
        assert data == data1_plus_2
        assert data != data1_c
        assert data2 == data2_c
