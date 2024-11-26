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
    """
    Test the cell count analysis pipeline with TMEV data

    Returns:
        _type_: _description_
    """

    @staticmethod
    @pytest.fixture(name="fpath_tmev_expected", scope="class")
    def fixture_fpath_tmev_expected(data_folder_tmev):
        """
        The file path of the expected TMEV results output file

        Args:
            data_folder_tmev (_type_): _description_

        Returns:
            _type_: _description_
        """
        return os.path.join(data_folder_tmev, "cell_count_pre_post_tmev_expected.xlsx")

    @staticmethod
    def test_fpath_tmev_expected_exists(fpath_tmev_expected):
        """
        Test if the expected TMEV results output file exists

        Args:
            fpath_tmev_expected (_type_): _description_
        """
        assert os.path.exists(fpath_tmev_expected)

    @staticmethod
    @pytest.fixture(name="df_expected_tmev", scope="class")
    def fixture_df_expected_tmev(fpath_tmev_expected):
        """
        The expected TMEV results output file as a dataframe

        Args:
            fpath_tmev_expected (_type_): _description_

        Returns:
            _type_: _description_
        """
        return pd.read_excel(fpath_tmev_expected)

    @staticmethod
    def test_df_expected_tmev_loaded(df_expected_tmev):
        """
        Test if the expected TMEV results output file is loaded as a dataframe

        Args:
            df_expected_tmev (_type_): _description_
        """
        assert isinstance(df_expected_tmev, pd.DataFrame)
        assert not df_expected_tmev.empty

    @staticmethod
    @pytest.fixture(name="json_tmev", scope="class")
    def fixture_json_tmev(data_folder):
        """The json file with TMEV data
        Args:
            data_folder_tmev (str): _description_
        Returns:
            str: _description_
        """
        return os.path.join(data_folder, "files_for_analysis_pre_post_tmev.json")

    @staticmethod
    @pytest.fixture(name="dict_input_files_tmev", scope="class")
    def fixture_dict_input_files_tmev(json_tmev):
        """
        Load the input files for the TMEV analysis

        Args:
            json_tmev (_type_): _description_

        Returns:
            _type_: _description_
        """
        return load_pre_post_files(json_tmev, os.path.dirname(json_tmev))

    @staticmethod
    def test_load_pre_post_files_tmev(dict_input_files_tmev):
        """
        Test the load_pre_post_files function on the input files for the TMEV analysis

        Args:
            dict_input_files_tmev (_type_): _description_
        """
        assert isinstance(dict_input_files_tmev, dict)
        for _, dict_exp_types in dict_input_files_tmev.items():
            for _, dict_files in dict_exp_types.items():
                assert "pre" in dict_files
                assert "post" in dict_files
                pre_files = dict_files["pre"]
                post_files = dict_files["post"]
                assert isinstance(pre_files, list)
                assert isinstance(post_files, list)
                assert len(pre_files) == len(post_files)
                for pre_file in pre_files:
                    assert os.path.exists(pre_file)
                    assert os.path.splitext(pre_file)[1] == ".hdf5"
                for post_file in post_files:
                    assert os.path.exists(post_file)
                    assert os.path.splitext(post_file)[1] == ".hdf5"

    @staticmethod
    def test_valid_json_tmev(json_tmev, data_folder_tmev):
        """Test if the json file with TMEV data is valid. Contents are also tested, assuming the
        json file is located at the root folder, and the file paths in the json file are relative
        to this root folder.
        Args:
            json_tmev (str): _description_
        """
        assert os.path.exists(json_tmev)
        assert os.path.isfile(json_tmev)
        assert os.path.splitext(json_tmev)[1] == ".json"
        dict_json = load_pre_post_files(json_tmev, os.path.dirname(json_tmev))
        for _, dict_exp_types in dict_json.items():
            for _, dict_files in dict_exp_types.items():
                assert "pre" in dict_files
                assert "post" in dict_files
                pre_files = [
                    os.path.join(data_folder_tmev, rel_path)
                    for rel_path in dict_files["pre"]
                ]
                post_files = [
                    os.path.join(data_folder_tmev, rel_path)
                    for rel_path in dict_files["post"]
                ]
                for pre_file in pre_files:
                    assert os.path.exists(pre_file)
                    assert os.path.splitext(pre_file)[1] == ".hdf5"
                for post_file in post_files:
                    assert os.path.exists(post_file)
                    assert os.path.splitext(post_file)[1] == ".hdf5"

    @staticmethod
    def test_main_tmev(dict_input_files_tmev, df_expected_tmev):
        """Test the main function with TMEV data
        Args:
            json_tmev (str): _description_
        """
        df_results = main(dict_input_files_tmev, save_results=False, output_folder=None)
        assert isinstance(df_results, pd.DataFrame)
        assert dataframes_equal(df_results, df_expected_tmev, both_nan_equal=True)


class TestCellCountAnalysisStim:
    """
    Test the cell count analysis pipeline with stim data

    Returns:
        _type_: _description_
    """

    @staticmethod
    @pytest.fixture(name="json_stim", scope="class")
    def fixture_json_stim(data_folder):
        """The json file with stim data
        Args:
            data_folder_stim (str): _description_
        Returns:
            str: _description_
        """
        return os.path.join(data_folder, "files_for_analysis_pre_post_stim.json")

    @staticmethod
    def test_valid_json_stim(json_stim, data_folder_stim):
        """Test if the json file with stim data is valid. Contents are also tested, assuming the
        json file is located at the root folder, and the file paths in the json file are relative
        to this root folder.
        Args:
            json_stim (str): _description_
        """
        assert os.path.exists(json_stim)
        assert os.path.isfile(json_stim)
        assert os.path.splitext(json_stim)[1] == ".json"
        dict_json = load_pre_post_files(json_stim, os.path.dirname(json_stim))
        for _, dict_exp_types in dict_json.items():
            for _, dict_files in dict_exp_types.items():
                assert "pre" in dict_files
                assert "post" in dict_files
                pre_files = [
                    os.path.join(data_folder_stim, rel_path)
                    for rel_path in dict_files["pre"]
                ]
                post_files = [
                    os.path.join(data_folder_stim, rel_path)
                    for rel_path in dict_files["post"]
                ]
                for pre_file in pre_files:
                    assert os.path.exists(pre_file)
                    assert os.path.splitext(pre_file)[1] == ".hdf5"
                for post_file in post_files:
                    assert os.path.exists(post_file)
                    assert os.path.splitext(post_file)[1] == ".hdf5"

    @staticmethod
    @pytest.fixture(name="fpath_stim_expected", scope="class")
    def fixture_fpath_stim_expected(data_folder_stim):
        """The file path of the expected stim results output file

        Args:
            data_folder_stim (_type_): _description_

        Returns:
            str: _description_
        """
        return os.path.join(data_folder_stim, "cell_count_pre_post_stim_expected.xlsx")

    @staticmethod
    def test_fpath_stim_expected_exists(fpath_stim_expected):
        """
        Test if the expected stim results output file exists

        Args:
            fpath_stim_expected (_type_): _description_
        """
        assert os.path.exists(fpath_stim_expected)

    @staticmethod
    @pytest.fixture(name="df_expected_stim", scope="class")
    def fixture_df_expected_stim(fpath_stim_expected):
        """
        The expected stim results output file as a dataframe

        Args:
            fpath_stim_expected (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        return pd.read_excel(fpath_stim_expected)

    @staticmethod
    def test_df_expected_stim_loaded(df_expected_stim):
        """
        Test if the expected stim results output file is loaded as a dataframe

        Args:
            df_expected_stim (_type_): _description_
        """
        assert isinstance(df_expected_stim, pd.DataFrame)
        assert not df_expected_stim.empty

    @staticmethod
    @pytest.fixture(name="dict_input_files_stim", scope="class")
    def fixture_dict_input_files_stim(json_stim):
        """
        Load the input files for the stim analysis

        Args:
            json_stim (_type_): _description_

        Returns:
            _type_: _description_
        """
        return load_pre_post_files(json_stim, os.path.dirname(json_stim))

    @staticmethod
    def test_load_pre_post_files_stim(dict_input_files_stim):
        """
        Test the load_pre_post_files function on the input files for the stim analysis

        Args:
            dict_input_files_stim (_type_): _description_
        """
        assert isinstance(dict_input_files_stim, dict)
        for _, dict_exp_types in dict_input_files_stim.items():
            for _, dict_files in dict_exp_types.items():
                assert "pre" in dict_files
                assert "post" in dict_files
                pre_files = dict_files["pre"]
                post_files = dict_files["post"]
                assert isinstance(pre_files, list)
                assert isinstance(post_files, list)
                assert len(pre_files) == len(post_files)
                for pre_file in pre_files:
                    assert os.path.exists(pre_file)
                    assert os.path.splitext(pre_file)[1] == ".hdf5"
                for post_file in post_files:
                    assert os.path.exists(post_file)
                    assert os.path.splitext(post_file)[1] == ".hdf5"

    @staticmethod
    def test_main_stim(dict_input_files_stim, df_expected_stim):
        """Test the main function with stim data
        Args:
            json_stim (str): _description_
        """
        df_results = main(dict_input_files_stim, save_results=False, output_folder=None)
        assert isinstance(df_results, pd.DataFrame)
        assert dataframes_equal(df_results, df_expected_stim, both_nan_equal=True)
