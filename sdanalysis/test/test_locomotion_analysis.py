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


def test_data_folder_exists(data_folder):
    assert os.path.exists(data_folder)

# Window stimulation (ChR2, jrgeco) data


@pytest.fixture(scope="module")
def data_folder_win_stim(data_folder):
    return os.path.join(data_folder, "Window stimulation")


def test_data_folder_win_stim_exists(data_folder_win_stim):
    assert os.path.exists(data_folder_win_stim)


@pytest.fixture(scope="module")
def loco_win_stim_fpath(data_folder_win_stim):
    return os.path.join(data_folder_win_stim, "loco_window-stim_output.xlsx")


@pytest.fixture(scope="module")
def loco_win_stim_delta_fpath(data_folder_win_stim):
    return os.path.join(data_folder_win_stim, "loco_window-stim_delta_output.xlsx")


@pytest.fixture(scope="module")
def loco_aggregate_win_stim_fpath(data_folder_win_stim):
    return os.path.join(data_folder_win_stim, "loco_window-stim_aggregate_output.xlsx")


@pytest.fixture(scope="module")
def loco_aggregate_delta_win_stim_fpath(data_folder_win_stim):
    return os.path.join(data_folder_win_stim, "loco_window-stim_aggregate_delta_output.xlsx")


@pytest.fixture(scope="module")
def loco_win_stim_traces_fpath(data_folder_win_stim):
    return os.path.join(data_folder_win_stim, "assembled_traces_window-stim.h5")


def test_loco_win_stim_files_exist(loco_win_stim_fpath, loco_win_stim_delta_fpath, loco_aggregate_win_stim_fpath, loco_aggregate_delta_win_stim_fpath, loco_win_stim_traces_fpath):
    assert os.path.exists(loco_win_stim_fpath)
    assert os.path.exists(loco_win_stim_delta_fpath)
    assert os.path.exists(loco_aggregate_win_stim_fpath)
    assert os.path.exists(loco_aggregate_delta_win_stim_fpath)
    assert os.path.exists(loco_win_stim_traces_fpath)


@pytest.fixture(scope="module")
def loco_win_stim_df(loco_win_stim_fpath):
    return pd.read_excel(loco_win_stim_fpath)


@pytest.fixture(scope="module")
def loco_win_stim_delta_df(loco_win_stim_delta_fpath):
    return pd.read_excel(loco_win_stim_delta_fpath)


@pytest.fixture(scope="module")
def loco_aggregate_win_stim(loco_aggregate_win_stim_fpath):
    return pd.read_excel(loco_aggregate_win_stim_fpath)


@pytest.fixture(scope="module")
def loco_aggregate_delta_win_stim(loco_aggregate_delta_win_stim_fpath):
    return pd.read_excel(loco_aggregate_delta_win_stim_fpath)


def test_locomotion_analysis_win_stim_results(loco_win_stim_df, loco_aggregate_win_stim, loco_win_stim_delta_df, loco_aggregate_delta_win_stim, loco_win_stim_traces_fpath):
    dfs = main(fpath=loco_win_stim_traces_fpath, save_data=False)
    assert isinstance(dfs, tuple)
    assert len(dfs) == 4
    # TODO: make utility function to compare dataframes (with bools NaN stuff), use it to compare with should-be output
    df, df_aggregate, df_delta, df_delta_aggregate = dfs
    # TODO: need to implement approximate equality comparison for floats.
    df.to_excel("D:\\Downloads\\loco_window-stim_output.xlsx")
    loco_win_stim_df.to_excel(
        "D:\\Downloads\\loco_window-stim_output_expected.xlsx")
    assert dataframes_equal(df, loco_win_stim_df, both_nan_equal=True)
    assert dataframes_equal(
        df_aggregate, loco_aggregate_win_stim, both_nan_equal=True)
    assert dataframes_equal(
        df_delta, loco_win_stim_delta_df, both_nan_equal=True)
    assert dataframes_equal(
        df_delta_aggregate, loco_aggregate_delta_win_stim, both_nan_equal=True)

# Cannula stimulation data


@pytest.fixture(scope="module")
def data_folder_cannula_stim(data_folder):
    return os.path.join(data_folder, "Cannula stimulation")


def test_data_folder_cannula_stim_exists(data_folder_cannula_stim):
    assert os.path.exists(data_folder_cannula_stim)


@pytest.fixture(scope="module")
def loco_cannula_stim_fpath(data_folder_cannula_stim):
    return os.path.join(data_folder_cannula_stim, "loco_cannula-stim_output.xlsx")


@pytest.fixture(scope="module")
def loco_cannula_stim_delta_fpath(data_folder_cannula_stim):
    return os.path.join(data_folder_cannula_stim, "loco_cannula-stim_delta_output.xlsx")


@pytest.fixture(scope="module")
def loco_aggregate_cannula_stim_fpath(data_folder_cannula_stim):
    return os.path.join(data_folder_cannula_stim, "loco_cannula-stim_aggregate_output.xlsx")


@pytest.fixture(scope="module")
def loco_aggregate_delta_cannula_stim_fpath(data_folder_cannula_stim):
    return os.path.join(data_folder_cannula_stim, "loco_cannula-stim_aggregate_delta_output.xlsx")


@pytest.fixture(scope="module")
def loco_cannula_stim_traces_fpath(data_folder_cannula_stim):
    return os.path.join(data_folder_cannula_stim, "assembled_traces_cannula-stim.h5")


def test_loco_cannula_stim_files_exist(loco_cannula_stim_fpath, loco_cannula_stim_delta_fpath, loco_aggregate_cannula_stim_fpath, loco_aggregate_delta_cannula_stim_fpath, loco_cannula_stim_traces_fpath):
    assert os.path.exists(loco_cannula_stim_fpath)
    assert os.path.exists(loco_cannula_stim_delta_fpath)
    assert os.path.exists(loco_aggregate_cannula_stim_fpath)
    assert os.path.exists(loco_aggregate_delta_cannula_stim_fpath)
    assert os.path.exists(loco_cannula_stim_traces_fpath)


@pytest.fixture(scope="module")
def loco_cannula_stim_df(loco_cannula_stim_fpath):
    return pd.read_excel(loco_cannula_stim_fpath)


@pytest.fixture(scope="module")
def loco_cannula_stim_delta_df(loco_cannula_stim_delta_fpath):
    return pd.read_excel(loco_cannula_stim_delta_fpath)


@pytest.fixture(scope="module")
def loco_aggregate_cannula_stim(loco_aggregate_cannula_stim_fpath):
    return pd.read_excel(loco_aggregate_cannula_stim_fpath)


@pytest.fixture(scope="module")
def loco_aggregate_delta_cannula_stim(loco_aggregate_delta_cannula_stim_fpath):
    return pd.read_excel(loco_aggregate_delta_cannula_stim_fpath)


def test_locomotion_analysis_cannula_stim_results(loco_cannula_stim_df, loco_aggregate_cannula_stim, loco_cannula_stim_delta_df, loco_aggregate_delta_cannula_stim, loco_cannula_stim_traces_fpath):
    dfs = main(fpath=loco_cannula_stim_traces_fpath, save_data=False)
    assert isinstance(dfs, tuple)
    assert len(dfs) == 4
    # TODO: make utility function to compare dataframes (with bools NaN stuff), use it to compare with should-be output
    df, df_aggregate, df_delta, df_delta_aggregate = dfs
    # TODO: need to implement approximate equality comparison for floats.
    assert dataframes_equal(df, loco_cannula_stim_df, both_nan_equal=True)
    assert dataframes_equal(
        df_aggregate, loco_aggregate_cannula_stim, both_nan_equal=True)
    assert dataframes_equal(
        df_delta, loco_cannula_stim_delta_df, both_nan_equal=True)
    assert dataframes_equal(
        df_delta_aggregate, loco_aggregate_delta_cannula_stim, both_nan_equal=True)


# TMEV data
@pytest.fixture(scope="module")
def data_folder_tmev(data_folder):
    return os.path.join(data_folder, "TMEV")


def test_data_folder_tmev_exists(data_folder_tmev):
    assert os.path.exists(data_folder_tmev)


@pytest.fixture(scope="module")
def loco_tmev_fpath(data_folder_tmev):
    return os.path.join(data_folder_tmev, "loco_tmev_output.xlsx")


@pytest.fixture(scope="module")
def loco_tmev_delta_fpath(data_folder_tmev):
    return os.path.join(data_folder_tmev, "loco_tmev_delta_output.xlsx")


@pytest.fixture(scope="module")
def loco_aggregate_tmev_fpath(data_folder_tmev):
    return os.path.join(data_folder_tmev, "loco_tmev_aggregate_output.xlsx")


@pytest.fixture(scope="module")
def loco_aggregate_delta_tmev_fpath(data_folder_tmev):
    return os.path.join(data_folder_tmev, "loco_tmev_aggregate_delta_output.xlsx")


@pytest.fixture(scope="module")
def loco_tmev_traces_fpath(data_folder_tmev):
    return os.path.join(data_folder_tmev, "assembled_traces_tmev.h5")


def test_loco_tmev_files_exist(loco_tmev_fpath, loco_tmev_delta_fpath, loco_aggregate_tmev_fpath, loco_aggregate_delta_tmev_fpath, loco_tmev_traces_fpath):
    assert os.path.exists(loco_tmev_fpath)
    assert os.path.exists(loco_tmev_delta_fpath)
    assert os.path.exists(loco_aggregate_tmev_fpath)
    assert os.path.exists(loco_aggregate_delta_tmev_fpath)
    assert os.path.exists(loco_tmev_traces_fpath)


@pytest.fixture(scope="module")
def loco_tmev_df(loco_tmev_fpath):
    return pd.read_excel(loco_tmev_fpath)


@pytest.fixture(scope="module")
def loco_tmev_delta_df(loco_tmev_delta_fpath):
    return pd.read_excel(loco_tmev_delta_fpath)


@pytest.fixture(scope="module")
def loco_aggregate_tmev(loco_aggregate_tmev_fpath):
    return pd.read_excel(loco_aggregate_tmev_fpath)


@pytest.fixture(scope="module")
def loco_aggregate_delta_tmev(loco_aggregate_delta_tmev_fpath):
    return pd.read_excel(loco_aggregate_delta_tmev_fpath)


def test_locomotion_analysis_tmev_results(loco_tmev_df, loco_aggregate_tmev, loco_tmev_delta_df, loco_aggregate_delta_tmev, loco_tmev_traces_fpath):
    dfs = main(fpath=loco_tmev_traces_fpath, save_data=False)
    assert isinstance(dfs, tuple)
    assert len(dfs) == 4
    # TODO: make utility function to compare dataframes (with bools NaN stuff), use it to compare with should-be output
    df, df_aggregate, df_delta, df_delta_aggregate = dfs
    # TODO: need to implement approximate equality comparison for floats.
    df.to_excel("D:\\Downloads\\df_loco.xlsx", index=False)
    loco_tmev_df.to_excel("D:\\Downloads\\df_loco_expected.xlsx", index=False)
    assert dataframes_equal(df, loco_tmev_df, both_nan_equal=True)
    assert dataframes_equal(
        df_aggregate, loco_aggregate_tmev, both_nan_equal=True)
    assert dataframes_equal(
        df_delta, loco_tmev_delta_df, both_nan_equal=True)
    assert dataframes_equal(
        df_delta_aggregate, loco_aggregate_delta_tmev, both_nan_equal=True)
