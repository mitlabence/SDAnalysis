"""
test_cell_count_analysis.py - Test the cell count analysis pipeline
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
    from cell_count_analysis import main, load_pre_post_files


@pytest.fixture(name="data_folder", scope="module")
def fixture_data_folder():
    """The root test data folder path
    Returns:
        str: _description_
    """
    env_dict = read_env()
    return os.path.join(env_dict["TEST_DATA_FOLDER"], "Cell_count_analysis")


@pytest.fixture(name="data_folder_tmev", scope="module")
def fixture_data_folder_tmev(data_folder):
    """The test data folder with TMEV data
    Args:
        data_folder (str): _description_
    Returns:
        str: _description_
    """
    return os.path.join(data_folder, "TMEV")


@pytest.fixture(name="data_folder_stim", scope="module")
def fixture_data_folder_stim(data_folder):
    """The test data folder with stim data
    Args:
        data_folder (str): _description_
    Returns:
        str: _description_
    """
    return os.path.join(data_folder, "stim")


def test_data_folder_exists(data_folder):
    """Test if the data folder exists
    Args:
        data_folder (str): _description_
    """
    assert os.path.exists(data_folder)


def test_data_folder_stim_exists(data_folder_stim):
    """Test if the stim data folder exists
    Args:
        data_folder_stim (str): _description_
    """
    assert os.path.exists(data_folder_stim)


def test_data_folder_tmev_exists(data_folder_tmev):
    """Test if the TMEV data folder exists
    Args:
        data_folder_tmev (str): _description_
    """
    assert os.path.exists(data_folder_tmev)


class TestCellCountAnalysisTMEV:
    @pytest.fixture(name="fpath_tmev_expected", scope="class")
    def fixture_fpath_tmev_expected(self, data_folder_tmev):
        return os.path.join(data_folder_tmev, "cell_count_pre_post_tmev_expected.xlsx")

    def test_fpath_tmev_expected_exists(self, fpath_tmev_expected):
        assert os.path.exists(fpath_tmev_expected)

    @pytest.fixture(name="df_expected_tmev", scope="class")
    def fixture_df_expected_tmev(self, fpath_tmev_expected):
        return pd.read_excel(fpath_tmev_expected)

    def test_df_expected_tmev_loaded(self, df_expected_tmev):
        assert isinstance(df_expected_tmev, pd.DataFrame)
        assert not df_expected_tmev.empty

    @pytest.fixture(name="json_tmev", scope="class")
    def fixture_json_tmev(self, data_folder_tmev):
        """The json file with TMEV data
        Args:
            data_folder_tmev (str): _description_
        Returns:
            str: _description_
        """
        return os.path.join(data_folder_tmev, "files_for_analysis_pre_post_tmev.json")

    def test_valid_json_tmev(self, json_tmev):
        """Test if the json file with TMEV data is valid
        Args:
            json_tmev (str): _description_
        """
        assert os.path.exists(json_tmev)
        assert os.path.isfile(json_tmev)
        assert os.path.splitext(json_tmev)[1] == ".json"
        dict_json = load_pre_post_files(json_tmev)
        for _, dict_exp_types in dict_json.items():
            for _, dict_files in dict_exp_types.items():
                assert "pre" in dict_files
                assert "post" in dict_files
                pre_files = dict_files["pre"]
                post_files = dict_files["post"]
                for pre_file in pre_files:
                    assert os.path.exists(pre_file)
                    assert os.path.splitext(pre_file)[1] == ".hdf5"
                for post_file in post_files:
                    assert os.path.exists(post_file)
                    assert os.path.splitext(post_file)[1] == ".hdf5"

    def test_main_tmev(self, json_tmev, df_expected_tmev):
        """Test the main function with TMEV data
        Args:
            json_tmev (str): _description_
        """
        df_results = main(json_tmev, save_results=False, output_folder=None)
        assert isinstance(df_results, pd.DataFrame)
        assert dataframes_equal(df_results, df_expected_tmev, both_nan_equal=True)


class TestCellCountAnalysisStim:
    @pytest.fixture(name="json_stim", scope="class")
    def fixture_json_stim(self, data_folder_stim):
        """The json file with stim data
        Args:
            data_folder_stim (str): _description_
        Returns:
            str: _description_
        """
        return os.path.join(data_folder_stim, "files_for_analysis_pre_post_stim.json")

    def test_valid_json_stim(self, json_stim):
        """Test if the json file with stim data is valid
        Args:
            json_stim (str): _description_
        """
        assert os.path.exists(json_stim)
        assert os.path.isfile(json_stim)
        assert os.path.splitext(json_stim)[1] == ".json"
        dict_json = load_pre_post_files(json_stim)
        for _, dict_exp_types in dict_json.items():
            for _, dict_files in dict_exp_types.items():
                assert "pre" in dict_files
                assert "post" in dict_files
                pre_files = dict_files["pre"]
                post_files = dict_files["post"]
                for pre_file in pre_files:
                    assert os.path.exists(pre_file)
                    assert os.path.splitext(pre_file)[1] == ".hdf5"
                for post_file in post_files:
                    assert os.path.exists(post_file)
                    assert os.path.splitext(post_file)[1] == ".hdf5"

    @pytest.fixture(name="fpath_stim_expected", scope="class")
    def fixture_fpath_stim_expected(self, data_folder_stim):
        return os.path.join(data_folder_stim, "cell_count_pre_post_stim_expected.xlsx")

    def test_fpath_stim_expected_exists(self, fpath_stim_expected):
        assert os.path.exists(fpath_stim_expected)

    @pytest.fixture(name="df_expected_stim", scope="class")
    def fixture_df_expected_stim(self, fpath_stim_expected):
        return pd.read_excel(fpath_stim_expected)

    def test_df_expected_stim_loaded(self, df_expected_stim):
        assert isinstance(df_expected_stim, pd.DataFrame)
        assert not df_expected_stim.empty

    def test_main_stim(self, json_stim, df_expected_stim):
        """Test the main function with stim data
        Args:
            json_stim (str): _description_
        """
        df_results = main(json_stim, save_results=False, output_folder=None)
        assert isinstance(df_results, pd.DataFrame)
        assert dataframes_equal(df_results, df_expected_stim, both_nan_equal=True)
