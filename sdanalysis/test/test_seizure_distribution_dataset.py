import os
import sys
import pandas as pd
import pytest

try:
    project_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )  # SDAnalysis/SDAnalysis folder
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )  # SDAnalysis folder (top-level folder)
    sys.path.insert(0, root_dir)
    sys.path.insert(0, project_dir)

except:
    raise Exception("Exception while adding root_dir to sys.path")
finally:
    from sdanalysis.env_reader import read_env
    from sdanalysis.seizure_distribution import *
    from sdanalysis.test.utils.dataframe_comparison import dataframes_equal



@pytest.fixture(name="path_sz_distribution", scope="module")
def fixture_sz_distribution_path():
    """
    The path to the seizure distribution files.

    Returns:
        str: the path to the seizure distribution files.
    """
    # from env_dict, read out the DATA_FOLDER
    env_dict = read_env()
    assert len(env_dict) > 0
    assert "DATA_FOLDER" in env_dict
    return os.path.join(env_dict["DATA_FOLDER"], "Seizure_distribution")

@pytest.fixture(name="fpath_sz_distribution_dset", scope="module")
def fixture_fpath_sz_distribution_dset(path_sz_distribution):
    """
    The path to the seizure distribution dataset (hdf5).

    Args:
        path_sz_distribution (str): the path to the seizure distribution files.

    Returns:
        str: the expected path to the seizure distribution file.
    """
    return os.path.join(path_sz_distribution, "tmev_seizures.hdf5")

@pytest.fixture(name="fpath_expected_sz_distribution", scope="module")
def fixture_fpath_expected_sz_distribution(path_sz_distribution):
    """
    The path to the expected seizure distribution sheet (xlsx).

    Args:
        path_sz_distribution (str): the path to the seizure distribution files.

    Returns:
        str: the expected path to the seizure distribution file.
    """
    return os.path.join(path_sz_distribution, "seizure_distribution_expected.xlsx")



def test_sz_distribution_files_exist(path_sz_distribution, fpath_sz_distribution_dset, fpath_expected_sz_distribution):
    """
    Test if the seizure distribution files exist.

    Args:
        path_sz_distribution (str): the path to the seizure distribution files.
    """
    assert os.path.exists(path_sz_distribution)
    # tests the files exist and are indeed files
    assert os.path.exists(fpath_sz_distribution_dset)
    assert os.path.isfile(fpath_sz_distribution_dset)
    assert os.path.exists(fpath_expected_sz_distribution)
    assert os.path.isfile(fpath_expected_sz_distribution)


@pytest.fixture(name="df_expected", scope="module")
def fixture_df_expected(fpath_expected_sz_distribution):
    """
    The expected seizure distribution (in hours after TMEV injection).

    Args:
        fpath_expected_sz_distribution (str): the path to the expected seizure distribution file (xlsx).

    Returns:
        pd.DataFrame: the expected seizure distribution DataFrame with each column representing a mouse, column names the mouse IDs, sorted by number of seizures (descending),
        and the entries in the rows representing the (ascending) time (in hours) of a seizure after injection.
    """
    df_expected = pd.read_excel(fpath_expected_sz_distribution, index_col=None)
    return df_expected

def test_sz_distribution_df(fpath_sz_distribution_dset, df_expected):
    """
    Test if the seizure distribution DataFrame matches the expected DataFrame.

    Args:
        fpath_sz_distribution_dset (str): the path to the seizure distribution dataset (hdf5).
        df_expected (pd.DataFrame): the expected seizure distribution DataFrame.
    """
    sz_times_dict_filtered, n_max_sz, n_mice = open_sz_distribution_file(fpath_sz_distribution_dset)
    df_actual = sz_distribution_df_short(sz_times_dict_filtered, n_max_sz, n_mice)
    # compare the two dataframes
    assert dataframes_equal(df_actual, df_expected, both_nan_equal=True, tolerance=0.01), "The actual seizure distribution DataFrame does not match the expected DataFrame."
