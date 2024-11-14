"""
test_directionality_analysis.py - Test the directionality analysis pipeline
"""
import os
import sys
import pytest
import pandas as pd
from utils.dataframe_comparison import dataframes_equal

try:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root_dir)
finally:
    from env_reader import read_env
    from directionality_analysis import main


@pytest.fixture(name="data_folder", scope="module")
def fixture_data_folder():
    """The test data folder path

    Returns:
        str: _description_
    """
    env_dict = read_env()
    return os.path.join(env_dict["TEST_DATA_FOLDER"], "Directionality_analysis")


@pytest.fixture(name="data_folder_stim", scope="module")
def fixture_data_folder_stim(data_folder):
    """The test data folder with stim data

    Args:
        data_folder (str): _description_

    Returns:
        str: _description_
    """
    return os.path.join(data_folder, "Used/ChR2+jrgeco")


@pytest.fixture(name="data_folder_tmev", scope="module")
def fixture_data_folder_tmev(data_folder):
    """The test data folder with TMEV data

    Args:
        data_folder (str): _description_

    Returns:
        str: _description_
    """
    return os.path.join(data_folder, "Used/TMEV")


@pytest.fixture(name="data_folder_tmev_ca1", scope="module")
def fixture_data_folder_tmev_ca1(data_folder_tmev):
    """The test data folder with TMEV data of CA1 windows

    Args:
        data_folder_tmev (str): _description_

    Returns:
        str: _description_
    """
    return os.path.join(data_folder_tmev, "CA1")


@pytest.fixture(name="data_folder_tmev_nc", scope="module")
def fixture_data_folder_tmev_nc(data_folder_tmev):
    """The test data folder with TMEV data of NC windows

    Args:
        data_folder_tmev (str): _description_

    Returns:
        str: _description_
    """
    return os.path.join(data_folder_tmev, "NC")


@pytest.fixture(name="fpath_df_angles_tmev_ca1", scope="module")
def fixture_fpath_df_angles_tmev_ca1(data_folder_tmev_ca1):
    return os.path.join(data_folder_tmev_ca1, "angles_tmev_ca1.xlsx")


@pytest.fixture(name="fpath_df_angles_aggregate_tmev_ca1", scope="module")
def fixture_fpath_df_angles_aggregate_tmev_ca1(data_folder_tmev_ca1):
    return os.path.join(data_folder_tmev_ca1, "angles_aggregate_tmev_ca1.xlsx")


def test_data_folder_exists(data_folder):
    assert os.path.exists(data_folder)


def test_data_folders_exist(
    data_folder_tmev, data_folder_tmev_ca1, data_folder_tmev_nc, data_folder_stim
):
    assert os.path.exists(data_folder_tmev)
    assert os.path.exists(data_folder_tmev_ca1)
    assert os.path.exists(data_folder_tmev_nc)
    assert os.path.exists(data_folder_stim)


def test_data_files_exist(fpath_df_angles_tmev_ca1, fpath_df_angles_aggregate_tmev_ca1):
    assert os.path.exists(fpath_df_angles_tmev_ca1)
    assert os.path.exists(fpath_df_angles_aggregate_tmev_ca1)


@pytest.fixture(name="df_angles_tmev_ca1", scope="module")
def fixture_df_angles_tmev_ca1(fpath_df_angles_tmev_ca1):
    return pd.read_excel(fpath_df_angles_tmev_ca1)


@pytest.fixture(name="df_angles_aggregate_tmev_ca1", scope="module")
def fixture_df_angles_aggregate_tmev_ca1(fpath_df_angles_aggregate_tmev_ca1):
    return pd.read_excel(fpath_df_angles_aggregate_tmev_ca1)


def test_directionality_analysis_tmev_ca1(
    data_folder_tmev_ca1, df_angles_tmev_ca1, df_angles_aggregate_tmev_ca1
):
    dfs = main(folder=data_folder_tmev_ca1, save_data=False)
    assert isinstance(dfs, tuple)
    assert len(dfs) == 2
    df_angles, df_angles_aggregate = dfs
    assert dataframes_equal(df_angles, df_angles_tmev_ca1, both_nan_equal=True)
    assert dataframes_equal(
        df_angles_aggregate, df_angles_aggregate_tmev_ca1, both_nan_equal=True
    )
