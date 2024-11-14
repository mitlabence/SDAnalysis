"""
test_speed_analysis.py - Test the speed analysis module
"""
import os
import sys
import pytest
import pandas as pd
from utils.dataframe_comparison import dataframes_equal

try:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, root_dir)
finally:
    from env_reader import read_env
    from speed_analysis import main


@pytest.fixture(name="data_folder", scope="module")
def fixture_data_folder():
    """The data folder for the test data

    Returns:
        str: _description_
    """
    env_dict = read_env()
    return os.path.join(env_dict["TEST_DATA_FOLDER"], "Directionality_analysis", "Used")


@pytest.fixture(name="fpath_df_mean_onset_speed_all", scope="module")
def fixture_fpath_df_mean_onset_speed_all(data_folder):
    """The path to the expected output file for the whole dataset

    Args:
        data_folder (str): _description_

    Returns:
        str: _description_
    """
    return os.path.join(data_folder, "mean_onset_speed_all.xlsx")


def test_data_folder_exists(data_folder):
    """Test whether the data folder exists

    Args:
        data_folder (str): _description_
    """
    assert os.path.exists(data_folder)


def test_data_files_exist(fpath_df_mean_onset_speed_all):
    """Test whether the files inside data folder exist

    Args:
        fpath_df_mean_onset_speed_all (str): _description_
    """
    assert os.path.exists(fpath_df_mean_onset_speed_all)


def test_speed_analysis_all(fpath_df_mean_onset_speed_all, data_folder):
    """Test the speed analysis output, compare with expected values

    Args:
        fpath_df_mean_onset_speed_all (pd.DataFrame): _description_
        data_folder (str): _description_
    """
    df_result = pd.read_excel(fpath_df_mean_onset_speed_all)
    dfs_list = main(folder=data_folder, save_data=False)
    assert isinstance(dfs_list, tuple)
    assert len(dfs_list) == 1
    df_mean_onset_speed_all = dfs_list[0]
    assert dataframes_equal(df_mean_onset_speed_all, df_result)
