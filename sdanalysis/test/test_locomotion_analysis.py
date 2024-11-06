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
    from locomotion_analysis import main


@pytest.fixture(scope="module")
def data_folder():
    env_dict = read_env()
    return os.path.join(env_dict["TEST_DATA_FOLDER"], "Locomotion_analysis")


@pytest.fixture(scope="module")
def loco_chr2_fpath(data_folder):
    return os.path.join(data_folder, "loco_chr2_output.xlsx")


@pytest.fixture(scope="module")
def loco_chr2_delta_fpath(data_folder):
    return os.path.join(data_folder, "loco_chr2_delta_output.xlsx")


@pytest.fixture(scope="module")
def loco_aggregate_chr2_fpath(data_folder):
    return os.path.join(data_folder, "loco_chr2_aggregate_output.xlsx")


@pytest.fixture(scope="module")
def loco_aggregate_delta_chr2_fpath(data_folder):
    return os.path.join(data_folder, "loco_chr2_aggregate_delta_output.xlsx")


@pytest.fixture(scope="module")
def loco_chr2_traces_fpath(data_folder):
    return os.path.join(data_folder, "assembled_traces_ChR2.h5")


def test_data_folder_exists(data_folder):
    assert os.path.exists(data_folder)


def test_loco_chr2_files_exist(loco_chr2_fpath, loco_chr2_delta_fpath, loco_aggregate_chr2_fpath, loco_aggregate_delta_chr2_fpath, loco_chr2_traces_fpath):
    assert os.path.exists(loco_chr2_fpath)
    assert os.path.exists(loco_chr2_delta_fpath)
    assert os.path.exists(loco_aggregate_chr2_fpath)
    assert os.path.exists(loco_aggregate_delta_chr2_fpath)
    assert os.path.exists(loco_chr2_traces_fpath)


@pytest.fixture(scope="module")
def loco_chr2_df(loco_chr2_fpath):
    return pd.read_excel(loco_chr2_fpath)


@pytest.fixture(scope="module")
def loco_chr2_delta_df(loco_chr2_delta_fpath):
    return pd.read_excel(loco_chr2_delta_fpath)


@pytest.fixture(scope="module")
def loco_aggregate_chr2(loco_aggregate_chr2_fpath):
    return pd.read_excel(loco_aggregate_chr2_fpath)


@pytest.fixture(scope="module")
def loco_aggregate_delta_chr2(loco_aggregate_delta_chr2_fpath):
    return pd.read_excel(loco_aggregate_delta_chr2_fpath)


def test_locomotion_analysis_chr2_results(loco_chr2_df, loco_aggregate_chr2, loco_chr2_delta_df, loco_aggregate_delta_chr2, loco_chr2_traces_fpath):
    dfs = main(fpath=loco_chr2_traces_fpath, save_data=False)
    assert isinstance(dfs, tuple)
    assert len(dfs) == 4
    # TODO: make utility function to compare dataframes (with bools NaN stuff), use it to compare with should-be output
    df, df_aggregate, df_delta, df_delta_aggregate = dfs
    # TODO: need to implement approximate equality comparison for floats.
    assert dataframes_equal(df, loco_chr2_df, both_nan_equal=True)
    assert dataframes_equal(
        df_aggregate, loco_aggregate_chr2, both_nan_equal=True)
    assert dataframes_equal(df_delta, loco_chr2_delta_df, both_nan_equal=True)
    assert dataframes_equal(
        df_delta_aggregate, loco_aggregate_delta_chr2, both_nan_equal=True)
