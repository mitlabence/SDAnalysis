"""
test_linear_motion.py - Test the following (coupled) modules: 
nd2_time_stamps.py, lv_time_stamps.py, lv_data.py, linear_motion.py
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
    from nd2_time_stamps import ND2TimeStamps
    from lv_data import LabViewData
    from lv_time_stamps import LabViewTimeStamps
    from linear_motion import LinearMotion


@pytest.fixture(name="data_folder_stim", scope="module")
def fixture_data_folder_stim():
    """
    The root directory of the test data

    Returns:
        _type_: _description_
    """
    env_dict = read_env()
    return os.path.join(env_dict["TEST_DATA_FOLDER"], "example_with_lfp")


def test_data_folder_stim_exists(data_folder_stim):
    """
    Test that the stim data folder exists

    Args:
        data_folder (_type_): _description_
    """
    assert os.path.exists(data_folder_stim)


@pytest.fixture(name="fpath_nik_time_stamps_stim", scope="module")
def fixture_fpath_nik_time_stamps_stim(data_folder_stim):
    """
    The file path of the stim recording

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    return os.path.join(data_folder_stim, "T370_ChR2_d29_elec_002_nik.txt")


def test_fpath_nik_time_stamps_stim_exists(fpath_nik_time_stamps_stim):
    """
    Test that the stim recording file exists

    Args:
        file_path_stim (_type_): _description_
    """
    assert os.path.exists(fpath_nik_time_stamps_stim)


@pytest.fixture(name="fpath_expected_nik_time_stamps_stim", scope="module")
def fixture_fpath_expected_nik_time_stamps_stim(data_folder_stim):
    """
    The file path of the expected time stamps for the stim recording

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    return os.path.join(data_folder_stim, "expected_nik_timestamps.xlsx")


def test_file_path_stim_exists(fpath_expected_nik_time_stamps_stim):
    """
    Test that the stim recording file exists

    Args:
        file_path_stim (_type_): _description_
    """
    assert os.path.exists(fpath_expected_nik_time_stamps_stim)


def test_fpath_expected_time_stamps_stim_exists(fpath_expected_nik_time_stamps_stim):
    """
    Test that the expected time stamps file exists

    Args:
        fpath_expected_nik_time_stamps_stim (_type_): _description_
    """
    assert os.path.exists(fpath_expected_nik_time_stamps_stim)


@pytest.fixture(name="expected_nik_time_stamps_stim", scope="module")
def fixture_expected_time_stamps_stim(fpath_expected_nik_time_stamps_stim):
    """
    The expected time stamps for the stim recording

    Args:
        fpath_expected_time_stamps (_type_): _description_

    Returns:
        _type_: _description_
    """
    return pd.read_excel(fpath_expected_nik_time_stamps_stim, header=0)

class TestND2TimeStamps:
    """
    Tests for the ND2TimeStamps class
    """
    def test_nd2_time_stamps(self, fpath_nik_time_stamps_stim, expected_nik_time_stamps_stim):
        """
        Test the ND2TimeStamps class with stim recording

        Args:
            data_folder (_type_): _description_
        """
        assert isinstance(fpath_nik_time_stamps_stim, str)
        nd2_time_stamps = ND2TimeStamps(fpath_nik_time_stamps_stim, "utf-16", "\t")
        assert isinstance(nd2_time_stamps.time_stamps, pd.DataFrame)
        assert dataframes_equal(
            nd2_time_stamps.time_stamps,
            expected_nik_time_stamps_stim,
        )


@pytest.fixture(name="fpath_lv_time_stamps_stim", scope="module")
def fixture_fpath_lv_time_stamps_stim(data_folder_stim):
    """
    The file path of the time stamps for the stim recording

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    return os.path.join(data_folder_stim, "T370_ChR2_d29_elec.050821.1729time.txt")


def test_fpath_lv_time_stamps_stim_exists(fpath_lv_time_stamps_stim):
    """
    Test that the time stamps file exists

    Args:
        file_path_time_stamps_stim (_type_): _description_
    """
    assert os.path.exists(fpath_lv_time_stamps_stim)


@pytest.fixture(name="fpath_expected_lv_time_stamps_stim", scope="module")
def fixture_fpath_expected_lv_time_stamps_stim(data_folder_stim):
    """
    The file path of the expected time stamps for the stim recording

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    return os.path.join(data_folder_stim, "lv_time_stamps_expected_after_opening.xlsx")


@pytest.fixture(name="expected_lv_time_stamps_stim", scope="module")
def fixture_expected_lv_time_stamps_stim(fpath_expected_lv_time_stamps_stim):
    """
    The expected time stamps for the stim recording

    Args:
        fpath_expected_time_stamps (_type_): _description_

    Returns:
        _type_: _description_
    """
    return pd.read_excel(fpath_expected_lv_time_stamps_stim, header=None)

class TestLabViewTimeStamps:
    """
    Tests for the LabViewTimeStamps class
    """
    def test_lv_time_stamps(self, fpath_lv_time_stamps_stim, expected_lv_time_stamps_stim):
        """
        Test the LabViewTimeStamps class with stim recording

        Args:
            data_folder (_type_): _description_
        """
        assert isinstance(fpath_lv_time_stamps_stim, str)
        lv_time_stamps = LabViewTimeStamps(fpath_lv_time_stamps_stim, "utf-8", "\t")
        assert isinstance(lv_time_stamps.time_stamps, pd.DataFrame)
        assert len(lv_time_stamps.time_stamps) == len(expected_lv_time_stamps_stim)
        assert dataframes_equal(
            lv_time_stamps.time_stamps,
            expected_lv_time_stamps_stim,
        )


@pytest.fixture(name="fpath_lv_data_stim", scope="module")
def fixture_fpath_lv_data_stim(data_folder_stim):
    """
    The file path of the data for the stim recording

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    return os.path.join(data_folder_stim, "T370_ChR2_d29_elec.050821.1729.txt")

def test_fpath_lv_data_stim_exists(fpath_lv_data_stim):
    """
    Test that the data file exists

    Args:
        file_path_data_stim (_type_): _description_
    """
    assert os.path.exists(fpath_lv_data_stim)

@pytest.fixture(name="fpath_expected_lv_data_stim", scope="module")
def fixture_fpath_expected_lv_data_stim(data_folder_stim):
    """
    The file path of the expected data for the stim recording

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    return os.path.join(data_folder_stim, "lv_data_expected_after_opening.xlsx")

def test_fpath_expected_lv_data_stim_exists(fpath_expected_lv_data_stim):
    """
    Test that the expected data file exists

    Args:
        fpath_expected_lv_data_stim (_type_): _description_
    """
    assert os.path.exists(fpath_expected_lv_data_stim)

@pytest.fixture(name="expected_lv_data_stim", scope="module")
def fixture_expected_lv_data_stim(fpath_expected_lv_data_stim):
    """
    The expected data for the stim recording

    Args:
        fpath_expected_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    return pd.read_excel(fpath_expected_lv_data_stim, header=None)

class TestLabViewData:
    """
    Tests for the LabViewData class
    """
    def test_lv_data(self, fpath_lv_data_stim, expected_lv_data_stim):
        """
        Test the LabViewData class with stim recording

        Args:
            data_folder (_type_): _description_
        """
        assert isinstance(fpath_lv_data_stim, str)
        lv_data = LabViewData(fpath_lv_data_stim, "utf-8", "\t")
        assert isinstance(lv_data.data, pd.DataFrame)
        assert len(lv_data.data) == len(expected_lv_data_stim)
        assert dataframes_equal(
            lv_data.data,
            expected_lv_data_stim,
        )
