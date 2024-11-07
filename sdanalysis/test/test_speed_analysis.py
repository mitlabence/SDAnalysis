import pytest
import pandas as pd
import os
import sys
from utils.dataframe_comparison import dataframes_equal

try:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, root_dir)
finally:
    from env_reader import read_env
    from speed_analysis import main


@pytest.fixture(scope="module")
def data_folder():
    env_dict = read_env()
    return os.path.join(env_dict["TEST_DATA_FOLDER"], "Directionality_analysis", "Used")


@pytest.fixture(scope="module")
def fpath_df_mean_onset_speed_all(data_folder):
    return os.path.join(data_folder, "mean_onset_speed_all.xlsx")


def test_data_folder_exists(data_folder):
    assert os.path.exists(data_folder)


def test_data_files_exist(fpath_df_mean_onset_speed_all):
    assert os.path.exists(fpath_df_mean_onset_speed_all)


def test_speed_analysis_all(fpath_df_mean_onset_speed_all, data_folder):
    df_result = pd.read_excel(fpath_df_mean_onset_speed_all)
    dfs = main(folder=data_folder, save_data=False)
    assert isinstance(dfs, tuple)
    assert len(dfs) == 1
    df_mean_onset_speed_all = dfs[0]
    assert dataframes_equal(df_mean_onset_speed_all, df_result)
