import os
import sys
import pytest
import pandas as pd
import numpy as np
from utils.dataframe_comparison import dataframes_equal

try:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root_dir)
finally:
    from env_reader import read_env
    from locomotion_analysis import main


@pytest.fixture(name="data_folder", scope="module")
def fixture_data_folder():
    """The data folder for the locomotion analysis tests

    Returns:
        str: The path
    """
    env_dict = read_env()
    return os.path.join(env_dict["TEST_DATA_FOLDER"], "Locomotion_analysis")


def test_data_folder_exists(data_folder):
    """Check if the data folder exists

    Args:
        data_folder (str): The data folder
    """
    assert os.path.exists(data_folder)


# Window stimulation (ChR2, jrgeco) data


@pytest.fixture(name="data_folder_win_stim", scope="module")
def fixture_data_folder_win_stim(data_folder):
    """Get the data folder for window stim group

    Args:
        data_folder (str): _description_

    Returns:
        str: _description_
    """
    return os.path.join(data_folder, "Window_stimulation")


def test_data_folder_win_stim_exists(data_folder_win_stim):
    """Check if the data folder for window stimulation exists

    Args:
        data_folder_win_stim (str): _description_
    """
    assert os.path.exists(data_folder_win_stim)


@pytest.fixture(name="loco_win_stim_fpath", scope="module")
def fixture_loco_win_stim_fpath(data_folder_win_stim):
    """Get the file path for the expected output after analyzing window stimulation data

    Args:
        data_folder_win_stim (str): _description_

    Returns:
        _type_: _description_
    """
    return os.path.join(data_folder_win_stim, "loco_window-stim_output.xlsx")


@pytest.fixture(name="loco_win_stim_delta_fpath", scope="module")
def fixture_loco_win_stim_delta_fpath(data_folder_win_stim):
    """The file path for the expected output delta values after analyzing window stimulation data

    Args:
        data_folder_win_stim (str): _description_

    Returns:
        str: _description_
    """
    return os.path.join(data_folder_win_stim, "loco_window-stim_delta_output.xlsx")


@pytest.fixture(name="loco_aggregate_win_stim_fpath", scope="module")
def fixture_loco_aggregate_win_stim_fpath(data_folder_win_stim):
    """The file path for the expected aggregate output after analyzing window stimulation data

    Args:
        data_folder_win_stim (str): _description_

    Returns:
        str: _description_
    """
    return os.path.join(data_folder_win_stim, "loco_window-stim_aggregate_output.xlsx")


@pytest.fixture(name="loco_aggregate_delta_win_stim_fpath", scope="module")
def fixture_loco_aggregate_delta_win_stim_fpath(data_folder_win_stim):
    """The file path for the expected aggregate delta output after analyzing window stimulation
    data

    Args:
        data_folder_win_stim (_type_): _description_

    Returns:
        _type_: _description_
    """
    return os.path.join(
        data_folder_win_stim, "loco_window-stim_aggregate_delta_output.xlsx"
    )


@pytest.fixture(name="loco_win_stim_traces_fpath", scope="module")
def fixture_loco_win_stim_traces_fpath(data_folder_win_stim):
    """The file path for the traces file for window stimulation data"""
    return os.path.join(data_folder_win_stim, "assembled_traces_window-stim.h5")


def test_loco_win_stim_files_exist(
    loco_win_stim_fpath,
    loco_win_stim_delta_fpath,
    loco_aggregate_win_stim_fpath,
    loco_aggregate_delta_win_stim_fpath,
    loco_win_stim_traces_fpath,
):
    """Test if the files for window stimulation data exist

    Args:
        loco_win_stim_fpath (_type_): _description_
        loco_win_stim_delta_fpath (_type_): _description_
        loco_aggregate_win_stim_fpath (_type_): _description_
        loco_aggregate_delta_win_stim_fpath (_type_): _description_
        loco_win_stim_traces_fpath (_type_): _description_
    """
    assert os.path.exists(loco_win_stim_fpath)
    assert os.path.exists(loco_win_stim_delta_fpath)
    assert os.path.exists(loco_aggregate_win_stim_fpath)
    assert os.path.exists(loco_aggregate_delta_win_stim_fpath)
    assert os.path.exists(loco_win_stim_traces_fpath)


@pytest.fixture(name="loco_win_stim_df", scope="module")
def fixture_loco_win_stim_df(loco_win_stim_fpath):
    """The dataframe for the expected output after analyzing window stimulation data

    Args:
        loco_win_stim_fpath (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    return pd.read_excel(loco_win_stim_fpath)


@pytest.fixture(name="loco_win_stim_delta_df", scope="module")
def fixture_loco_win_stim_delta_df(loco_win_stim_delta_fpath):
    """The dataframe for the expected delta values after analyzing window stimulation data

    Args:
        loco_win_stim_delta_fpath (_type_): _description_

    Returns:
        pd.DataFrame: _description_
    """
    return pd.read_excel(loco_win_stim_delta_fpath)


@pytest.fixture(name="loco_aggregate_win_stim_df", scope="module")
def fixture_loco_aggregate_win_stim_df(loco_aggregate_win_stim_fpath):
    """The dataframe for the expected aggregate output after analyzing window stimulation data

    Args:
        loco_aggregate_win_stim_fpath (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    return pd.read_excel(loco_aggregate_win_stim_fpath)


@pytest.fixture(name="loco_aggregate_delta_win_stim_df", scope="module")
def fixture_loco_aggregate_delta_win_stim_df(loco_aggregate_delta_win_stim_fpath):
    """
    The dataframe for the expected aggregate delta output after analyzing window stimulation data
    """
    return pd.read_excel(loco_aggregate_delta_win_stim_fpath)


def test_locomotion_analysis_win_stim_results(
    loco_win_stim_df,
    loco_aggregate_win_stim_df,
    loco_win_stim_delta_df,
    loco_aggregate_delta_win_stim_df,
    loco_win_stim_traces_fpath,
):
    dfs = main(fpath=loco_win_stim_traces_fpath, save_data=False)
    assert isinstance(dfs, tuple)
    assert len(dfs) == 4
    # TODO: make utility function to compare dataframes (with bools NaN stuff), use it to compare with should-be output
    df, df_aggregate, df_delta, df_delta_aggregate = dfs
    # TODO: need to implement approximate equality comparison for floats.
    assert dataframes_equal(df, loco_win_stim_df, both_nan_equal=True)
    assert dataframes_equal(
        df_aggregate, loco_aggregate_win_stim_df, both_nan_equal=True
    )
    assert dataframes_equal(df_delta, loco_win_stim_delta_df, both_nan_equal=True)
    assert dataframes_equal(
        df_delta_aggregate, loco_aggregate_delta_win_stim_df, both_nan_equal=True
    )


# Cannula stimulation data


@pytest.fixture(name="data_folder_cannula_stim", scope="module")
def fixture_data_folder_cannula_stim(data_folder):
    return os.path.join(data_folder, "Cannula_stimulation")


def test_data_folder_cannula_stim_exists(data_folder_cannula_stim):
    assert os.path.exists(data_folder_cannula_stim)


@pytest.fixture(name="loco_cannula_stim_fpath", scope="module")
def fixture_loco_cannula_stim_fpath(data_folder_cannula_stim):
    return os.path.join(data_folder_cannula_stim, "loco_cannula-stim_output.xlsx")


@pytest.fixture(name="loco_cannula_stim_delta_fpath", scope="module")
def fixture_loco_cannula_stim_delta_fpath(data_folder_cannula_stim):
    return os.path.join(data_folder_cannula_stim, "loco_cannula-stim_delta_output.xlsx")


@pytest.fixture(name="loco_aggregate_cannula_stim_fpath", scope="module")
def fixture_loco_aggregate_cannula_stim_fpath(data_folder_cannula_stim):
    return os.path.join(
        data_folder_cannula_stim, "loco_cannula-stim_aggregate_output.xlsx"
    )


@pytest.fixture(name="loco_aggregate_delta_cannula_stim_fpath", scope="module")
def fixture_loco_aggregate_delta_cannula_stim_fpath(data_folder_cannula_stim):
    return os.path.join(
        data_folder_cannula_stim, "loco_cannula-stim_aggregate_delta_output.xlsx"
    )


@pytest.fixture(name="loco_cannula_stim_traces_fpath", scope="module")
def fixture_loco_cannula_stim_traces_fpath(data_folder_cannula_stim):
    return os.path.join(data_folder_cannula_stim, "assembled_traces_cannula-stim.h5")


def test_loco_cannula_stim_files_exist(
    loco_cannula_stim_fpath,
    loco_cannula_stim_delta_fpath,
    loco_aggregate_cannula_stim_fpath,
    loco_aggregate_delta_cannula_stim_fpath,
    loco_cannula_stim_traces_fpath,
):
    assert os.path.exists(loco_cannula_stim_fpath)
    assert os.path.exists(loco_cannula_stim_delta_fpath)
    assert os.path.exists(loco_aggregate_cannula_stim_fpath)
    assert os.path.exists(loco_aggregate_delta_cannula_stim_fpath)
    assert os.path.exists(loco_cannula_stim_traces_fpath)


@pytest.fixture(name="loco_cannula_stim_df", scope="module")
def fixture_loco_cannula_stim_df(loco_cannula_stim_fpath):
    df = pd.read_excel(loco_cannula_stim_fpath)
    # window_type might contain "None" string, which is converted to np.NaN
    df["window_type"] = df["window_type"].replace(np.NaN, "None")
    return df


@pytest.fixture(name="loco_cannula_stim_delta_df", scope="module")
def fixture_loco_cannula_stim_delta_df(loco_cannula_stim_delta_fpath):
    df = pd.read_excel(loco_cannula_stim_delta_fpath)
    # window_type might contain "None" string, which is converted to np.NaN
    df["window_type"] = df["window_type"].replace(np.NaN, "None")
    return df


@pytest.fixture(name="loco_aggregate_cannula_stim", scope="module")
def fixture_loco_aggregate_cannula_stim(loco_aggregate_cannula_stim_fpath):
    df = pd.read_excel(loco_aggregate_cannula_stim_fpath)
    # window_type might contain "None" string, which is converted to np.NaN
    df["window_type"] = df["window_type"].replace(np.NaN, "None")
    return df


@pytest.fixture(name="loco_aggregate_delta_cannula_stim", scope="module")
def fixture_loco_aggregate_delta_cannula_stim(loco_aggregate_delta_cannula_stim_fpath):
    df = pd.read_excel(loco_aggregate_delta_cannula_stim_fpath)
    # window_type might contain "None" string, which is converted to np.NaN
    df["window_type"] = df["window_type"].replace(np.NaN, "None")
    return df


def test_locomotion_analysis_cannula_stim_results(
    loco_cannula_stim_df,
    loco_aggregate_cannula_stim,
    loco_cannula_stim_delta_df,
    loco_aggregate_delta_cannula_stim,
    loco_cannula_stim_traces_fpath,
):
    dfs = main(fpath=loco_cannula_stim_traces_fpath, save_data=False)
    assert isinstance(dfs, tuple)
    assert len(dfs) == 4
    # TODO: make utility function to compare dataframes (with bools NaN stuff), use it to compare with should-be output
    df, df_aggregate, df_delta, df_delta_aggregate = dfs
    # TODO: need to implement approximate equality comparison for floats.
    assert dataframes_equal(df, loco_cannula_stim_df, both_nan_equal=True)
    assert dataframes_equal(
        df_aggregate, loco_aggregate_cannula_stim, both_nan_equal=True
    )
    assert dataframes_equal(df_delta, loco_cannula_stim_delta_df, both_nan_equal=True)
    assert dataframes_equal(
        df_delta_aggregate, loco_aggregate_delta_cannula_stim, both_nan_equal=True
    )


# TMEV data
@pytest.fixture(name="data_folder_tmev", scope="module")
def fixture_data_folder_tmev(data_folder):
    return os.path.join(data_folder, "TMEV")


def test_data_folder_tmev_exists(data_folder_tmev):
    assert os.path.exists(data_folder_tmev)


@pytest.fixture(name="loco_tmev_fpath", scope="module")
def fixture_loco_tmev_fpath(data_folder_tmev):
    return os.path.join(data_folder_tmev, "loco_tmev_output.xlsx")


@pytest.fixture(name="loco_tmev_delta_fpath", scope="module")
def fixture_loco_tmev_delta_fpath(data_folder_tmev):
    return os.path.join(data_folder_tmev, "loco_tmev_delta_output.xlsx")


@pytest.fixture(name="loco_aggregate_tmev_fpath", scope="module")
def fixture_loco_aggregate_tmev_fpath(data_folder_tmev):
    return os.path.join(data_folder_tmev, "loco_tmev_aggregate_output.xlsx")


@pytest.fixture(name="loco_aggregate_delta_tmev_fpath", scope="module")
def fixture_loco_aggregate_delta_tmev_fpath(data_folder_tmev):
    return os.path.join(data_folder_tmev, "loco_tmev_aggregate_delta_output.xlsx")


@pytest.fixture(name="loco_tmev_traces_fpath", scope="module")
def fixture_loco_tmev_traces_fpath(data_folder_tmev):
    return os.path.join(data_folder_tmev, "assembled_traces_tmev.h5")


def test_loco_tmev_files_exist(
    loco_tmev_fpath,
    loco_tmev_delta_fpath,
    loco_aggregate_tmev_fpath,
    loco_aggregate_delta_tmev_fpath,
    loco_tmev_traces_fpath,
):
    assert os.path.exists(loco_tmev_fpath)
    assert os.path.exists(loco_tmev_delta_fpath)
    assert os.path.exists(loco_aggregate_tmev_fpath)
    assert os.path.exists(loco_aggregate_delta_tmev_fpath)
    assert os.path.exists(loco_tmev_traces_fpath)


@pytest.fixture(name="loco_tmev_df", scope="module")
def fixture_loco_tmev_df(loco_tmev_fpath):
    return pd.read_excel(loco_tmev_fpath)


@pytest.fixture(name="loco_tmev_delta_df", scope="module")
def fixture_loco_tmev_delta_df(loco_tmev_delta_fpath):
    return pd.read_excel(loco_tmev_delta_fpath)


@pytest.fixture(name="loco_aggregate_tmev_df", scope="module")
def fixture_loco_aggregate_tmev_df(loco_aggregate_tmev_fpath):
    return pd.read_excel(loco_aggregate_tmev_fpath)


@pytest.fixture(name="loco_aggregate_delta_tmev_df", scope="module")
def fixture_loco_aggregate_delta_tmev_df(loco_aggregate_delta_tmev_fpath):
    return pd.read_excel(loco_aggregate_delta_tmev_fpath)


def test_locomotion_analysis_tmev_results(
    loco_tmev_df,
    loco_aggregate_tmev_df,
    loco_tmev_delta_df,
    loco_aggregate_delta_tmev_df,
    loco_tmev_traces_fpath,
):
    dfs_list = main(fpath=loco_tmev_traces_fpath, save_data=True)
    assert isinstance(dfs_list, tuple)
    assert len(dfs_list) == 4
    # TODO: make utility function to compare dataframes (with bools NaN stuff), use it to compare with should-be output
    df_individual, df_aggregate, df_delta, df_delta_aggregate = dfs_list
    # TODO: need to implement approximate equality comparison for floats.
    assert dataframes_equal(df_individual, loco_tmev_df, both_nan_equal=True)
    assert dataframes_equal(df_aggregate, loco_aggregate_tmev_df, both_nan_equal=True)
    assert dataframes_equal(df_delta, loco_tmev_delta_df, both_nan_equal=True)
    assert dataframes_equal(
        df_delta_aggregate, loco_aggregate_delta_tmev_df, both_nan_equal=True
    )
