"""
test_linear_motion.py - Test the following (coupled) modules: 
nd2_time_stamps.py, lv_time_stamps.py, lv_data.py, linear_motion.py
"""

import os
import sys
import json
from typing import Tuple
import pytest
import pandas as pd
import numpy as np
from utils.dataframe_comparison import dataframes_equal, series_equal

try:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root_dir)
finally:
    from env_reader import read_env
    from nd2_time_stamps import ND2TimeStamps
    from lv_data import LabViewData
    from lv_time_stamps import LabViewTimeStamps
    from sdanalysis.linear_locomotion import LinearLocomotion

DICT_MATLAB_PYTHON_VARIABLES = {
    "speed": "speed",
    "distance": "total_distance",
    "round": "round",
    "distancePR": "distance_per_round",
    "reflect": "reflectivity",
    "stripes": "stripes_total",
    "stripesPR": "stripes_per_round",
    "time": "time_total_s",
    "timePR": "time_per_round",
}  # the matlab original parameter names and their corresponding python naming


@pytest.fixture(name="col_names_lv_data", scope="module")
def fixture_col_names_lv_data():
    """
    The columns of the LabView output data file (not the time stamps)

    Returns:
        _type_: _description_
    """
    return [
        "round",
        "speed",
        "total_distance",
        "distance_per_round",
        "reflectivity",
        "unknown",
        "stripes_total",
        "stripes_per_round",
        "time_total_ms",
        "time_per_round",
        "stimuli1",
        "stimuli2",
        "stimuli3",
        "stimuli4",
        "stimuli5",
        "stimuli6",
        "stimuli7",
        "stimuli8",
        "stimuli9",
        "pupil_area",
    ]


@pytest.fixture(name="col_names_lv_time_stamps", scope="module")
def fixture_col_names_lv_time_stamps():
    """
    The columns of the LabView time stamps file

    Returns:
        _type_: _description_
    """
    return [
        "belt_time_stamps",
        "resonant_time_stamps",
        "galvano_time_stamps",
        "unused",
    ]


@pytest.fixture(name="data_folder", scope="module")
def fixture_data_folder():
    """
    The root directory of the test data

    Returns:
        _type_: _description_
    """
    env_dict = read_env()
    return os.path.join(env_dict["TEST_DATA_FOLDER"], "Matching")


def test_data_folder_exists(data_folder):
    """
    Test that the stim data folder exists

    Args:
        data_folder (_type_): _description_
    """
    assert os.path.exists(data_folder)


@pytest.fixture(name="data_fpaths", scope="module")
def fixture_data_fpaths(data_folder):
    """
    The file paths to the matching test data

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    folder_names = [
        "lfp_stim_complete_session",
        "lfp_stim_decimal",
        "lfp_stim_nondecimal_1",
        "lfp_tmev_1",
        "lfp_tmev_2",
        "no_lfp_tmev_1",
        "no_lfp_tmev_2",
        "no_lfp_tmev_3",
        "no_lfp_tmev_4",
    ]
    fpaths = []
    for folder_name in folder_names:
        directory = os.path.join(data_folder, folder_name)
        fpaths.append(
            # nikon time stamps, labview data, labview time stamps,
            # expected formatted nikon time stamps, labview data, and labview time stamps files,
            # expected labview data after matching to Nikon,
            # expected "belt_struct" structure at various steps in the original matlab pipeline,
            # and "belt_struct" and "belt_scn" at the end of the pipeline
            # they have same base name as the folder
            (
                os.path.join(
                    directory, folder_name + "_nik.txt"
                ),  # the nikon time stamps file
                os.path.join(
                    directory, folder_name + ".txt"
                ),  # the labview (belt) data file
                # the labview time stamps (belt + scanner)
                os.path.join(directory, folder_name + "time.txt"),
                # the expected loaded + reformatted nikon time stamps
                os.path.join(directory, folder_name + "_nik_expected.xlsx"),
                # the expected loaded labview data
                os.path.join(directory, folder_name + "_expected_after_opening.hdf5"),
                # the expected loaded labview time stamps
                os.path.join(directory, folder_name + "time_expected.xlsx"),
                # the expected output parameters after matching to Nikon
                os.path.join(directory, folder_name + "_expected_after_match.json"),
                # the expected loaded labview data and scanner time stamps after matching to Nikon
                os.path.join(directory, folder_name + "_expected_after_match.hdf5"),
                os.path.join(
                    directory, folder_name + "_expected_after_arduino_corr.hdf5"
                ),
                os.path.join(directory, folder_name + "_expected_after_belt_corr.hdf5"),
                os.path.join(directory, folder_name + "_expected_after_pipeline.hdf5"),
            )
        )
    return fpaths


def test_data_fpaths(data_fpaths):
    """
    Test that the data file paths exist

    Args:
        data_fpaths (_type_): _description_
    """
    for fpath in data_fpaths:
        for file_path in fpath:
            assert os.path.exists(file_path)


@pytest.fixture(name="fpaths_nik_time_stamps", scope="module")
def fixture_fpaths_nik_time_stamps(data_fpaths):
    """
    The file paths of the stim recording nikon time stamps files.

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    return [data_fpath[0] for data_fpath in data_fpaths]


class TestND2TimeStamps:
    """
    Tests for the ND2TimeStamps class
    """

    @pytest.fixture(name="fpaths_expected_nik_time_stamps", scope="module")
    def fixture_fpath_expected_nik_time_stamps_stim(self, data_fpaths):
        """
        The file paths of the expected time stamps for each recording

        Args:
            data_folder (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [fpaths[3] for fpaths in data_fpaths]

    @pytest.fixture(name="expected_nik_time_stamps", scope="class")
    def fixture_expected_time_stamps_stim(self, fpaths_expected_nik_time_stamps):
        """
        The expected time stamps for the stim recording

        Args:
            fpath_expected_time_stamps (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [
            pd.read_excel(fpath, header=0) for fpath in fpaths_expected_nik_time_stamps
        ]

    def test_nd2_time_stamps(self, fpaths_nik_time_stamps, expected_nik_time_stamps):
        """
        Test the ND2TimeStamps class with stim recording

        Args:
            data_folder (_type_): _description_
        """
        for i, fpath in enumerate(fpaths_nik_time_stamps):
            assert isinstance(fpath, str)
            nd2_time_stamps = ND2TimeStamps(fpath, "utf-16", "\t")
            assert isinstance(nd2_time_stamps.time_stamps, pd.DataFrame)
            assert dataframes_equal(
                nd2_time_stamps.time_stamps,
                expected_nik_time_stamps[i],
            )


@pytest.fixture(name="fpaths_lv_time_stamps", scope="module")
def fixture_fpaths_lv_time_stamps(data_fpaths):
    """
    The file path of the time stamps for the stim recording

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    return [data_fpath[2] for data_fpath in data_fpaths]


class TestLabViewTimeStamps:
    """
    Tests for the LabViewTimeStamps class
    """

    @pytest.fixture(name="fpaths_expected_lv_time_stamps", scope="class")
    def fixture_fpaths_expected_lv_time_stamps(self, data_fpaths):
        """
        The file path of the expected time stamps for the stim recording

        Args:
            data_folder (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [data_fpath[5] for data_fpath in data_fpaths]

    @pytest.fixture(name="expected_lv_time_stamps", scope="class")
    def fixture_expected_lv_time_stamps(
        self, fpaths_expected_lv_time_stamps, col_names_lv_time_stamps
    ):
        """
        The expected time stamps for the stim recording

        Args:
            fpath_expected_time_stamps (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [
            pd.read_excel(fpath, header=None).rename(
                columns=dict(enumerate(col_names_lv_time_stamps))
            )
            for fpath in fpaths_expected_lv_time_stamps
        ]
        # FIXME: reading speed too slow, need hdf5!

    def test_lv_time_stamps(self, fpaths_lv_time_stamps, expected_lv_time_stamps):
        """
        Test the LabViewTimeStamps class with stim recording

        Args:
            data_folder (_type_): _description_
        """
        for i, expected_lv_time_stamp in enumerate(expected_lv_time_stamps):
            lv_time_stamps = LabViewTimeStamps(fpaths_lv_time_stamps[i], "utf-8", "\t")
            assert isinstance(expected_lv_time_stamp, pd.DataFrame)
            assert dataframes_equal(
                lv_time_stamps.time_stamps_ms,  # matlab code used to work with milliseconds
                expected_lv_time_stamp,
            )


@pytest.fixture(name="fpaths_lv_data", scope="module")
def fixture_fpaths_lv_data(data_fpaths):
    """
    The file path of the data for the stim recording

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    return [data_fpath[1] for data_fpath in data_fpaths]


class TestLabViewData:
    """
    Tests for the LabViewData class
    """

    @pytest.fixture(name="fpaths_expected_lv_data", scope="class")
    def fixture_fpath_expected_lv_data_stim(self, data_fpaths):
        """
        The file path of the expected data for the stim recording

        Args:
            data_folder (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [data_fpath[4] for data_fpath in data_fpaths]

    @pytest.fixture(name="expected_lv_data", scope="class")
    def fixture_expected_lv_data_stim(self, fpaths_expected_lv_data, col_names_lv_data):
        """
        The expected data for the stim recording

        Args:
            fpath_expected_data (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [
            pd.read_hdf(fpath, header=None).rename(
                columns=dict(enumerate(col_names_lv_data))
            )
            for fpath in fpaths_expected_lv_data
        ]

    def test_lv_data(self, fpaths_lv_data, expected_lv_data):
        """
        Test the LabViewData class with stim recording

        Args:
            data_folder (_type_): _description_
        """
        for i, fpath in enumerate(fpaths_lv_data):
            assert isinstance(fpath, str)
            lv_data = LabViewData(fpath, "utf-8", "\t")
            assert isinstance(lv_data.data, pd.DataFrame)
            assert isinstance(
                lv_data.data_ms, pd.DataFrame
            )  # matlab code used to work with milliseconds
            assert len(lv_data.data_ms) == len(expected_lv_data[i])
            assert dataframes_equal(
                lv_data.data_ms,
                expected_lv_data[i],
                both_nan_equal=True,
            )


@pytest.fixture(name="fpaths_expected_data_after_match", scope="module")
def fixture_fpaths_expected_data_after_match(data_fpaths):
    """
    The file path of the expected data after matching for the stim recording

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    return [data_fpath[6] for data_fpath in data_fpaths]


class TestMatchingBeltToScanner:
    """
    Tests for the matching of the belt recording to the scanner recording
    The processing has the following steps:
            matching LV time stamps to ND2 time stamps (after_matching tests up until this step)
            correcting arduino artifacts (after_arduino_corr tests up until this step)
            correcting belt length (after_belt_corr tests up until this step)
            adding binary running, correcting speed units (after_pipeline tests up until this step)
    """

    @pytest.fixture(name="fpaths_expected_params_after_match", scope="class")
    def fixture_fpaths_expected_params_after_matching(self, data_fpaths):
        """
        The file path of the expected parameters for the stim recording

        Args:
            data_folder (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [data_fpath[6] for data_fpath in data_fpaths]

    @pytest.fixture(name="fpaths_expected_belt_after_match", scope="class")
    def fixture_fpaths_expected_belt_after_matching(self, data_fpaths):
        """
        The file path of the expected time stamps for the belt recording

        Args:
            data_folder (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [data_fpath[7] for data_fpath in data_fpaths]

    @pytest.fixture(name="fpaths_expected_tsscn_after_match", scope="class")
    def fixture_fpaths_expected_tsscn_after_matching(self, data_fpaths):
        """
        The file path of the expected time stamps for the scanner recording

        Args:
            data_folder (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [data_fpath[8] for data_fpath in data_fpaths]

    @pytest.fixture(name="expected_tsscn_after_match", scope="class")
    def fixture_expected_tsscn_after_matching(self, fpaths_expected_belt_after_match):
        """
        The expected time stamps for the scanner recording

        Args:
            fpath_expected_time_stamps (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [
            pd.read_hdf(fpath, header=None, key="df_tsscn").rename(columns={0: "tsscn"})
            for fpath in fpaths_expected_belt_after_match
        ]

    @pytest.fixture(name="expected_params_after_match", scope="class")
    def fixture_expected_params_after_matching(
        self, fpaths_expected_params_after_match
    ):
        """
        The expected parameters for the stim recording

        Args:
            fpath_expected_params (_type_): _description_

        Returns:
            _type_: _description_
        """
        data_list = []
        for fpath in fpaths_expected_params_after_match:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                data_list.append(data)
        return data_list

    @pytest.fixture(name="expected_belt_after_match", scope="class")
    def fixture_expected_belt_after_matching(
        self, fpaths_expected_belt_after_match, col_names_lv_data
    ):
        """
        The expected time stamps for the belt recording

        Args:
            fpath_expected_time_stamps (_type_): _description_

        Returns:
            _type_: _description_
        """
        dfs = [
            pd.read_hdf(fpath, header=None, key="df").rename(
                columns=dict(enumerate(col_names_lv_data))
            )
            for fpath in fpaths_expected_belt_after_match
        ]
        for i, _ in enumerate(dfs):
            if "time_total_ms" in dfs[i].columns:
                dfs[i]["time_total_s"] = dfs[i]["time_total_ms"] / 1000.0
                dfs[i].drop(columns=["time_total_ms"], inplace=True)
                dfs[i]["time_per_round"] = dfs[i]["time_per_round"] / 1000.0
        return dfs

    def test_params_after_matching(
        self,
        fpaths_nik_time_stamps,
        fpaths_lv_time_stamps,
        fpaths_lv_data,
        expected_params_after_match,
    ):
        """
        Compare the parameters after matching LV with ND2 time stamps. The parameters do not
        include parameters of Arduino artifact removal, belt length correction, and other
        post-processing steps.



        Args:
            fpaths_nik_time_stamps (_type_): _description_
            fpaths_lv_time_stamps (_type_): _description_
            fpaths_lv_data (_type_): _description_
            expected_params_after_match (_type_): _description_
        """
        for i, fpath_nik in enumerate(fpaths_nik_time_stamps):
            nik_time_stamps = ND2TimeStamps(fpath_nik, "utf-16", "\t")
            lv_time_stamps = LabViewTimeStamps(fpaths_lv_time_stamps[i], "utf-8", "\t")
            lv_data = LabViewData(fpaths_lv_data[i], "utf-8", "\t")

            linear_locomotion = LinearLocomotion(
                nik_time_stamps, lv_time_stamps, lv_data, break_after_matching=True
            )
            expected_params = expected_params_after_match[i]
            assert isinstance(linear_locomotion.scanner_time_stamps, pd.Series)
            assert expected_params["missed_frames"] == linear_locomotion.n_missed_frames
            assert (
                expected_params["missed_belt_cycles"]
                == linear_locomotion.n_missed_cycles
            )
            assert (
                expected_params["source_timestamps"]
                == linear_locomotion.source_scanner_time_stamps
            )
            assert np.isclose(
                expected_params["duration"], linear_locomotion.duration, atol=0.0001
            )
            assert np.isclose(
                expected_params["frequency"],
                linear_locomotion.imaging_frequency,
                atol=0.0001,
            )
            # matlab has 1-indexing, correct for that
            assert expected_params["i_belt_start"] == linear_locomotion.i_belt_start + 1
            assert expected_params["i_belt_stop"] == linear_locomotion.i_belt_stop + 1

    def test_linear_locomotion_after_matching(
        self,
        fpaths_nik_time_stamps,
        fpaths_lv_time_stamps,
        fpaths_lv_data,
        expected_tsscn_after_match,
        expected_belt_after_match,
    ):
        """Test the LinearLocomotion class with all recordings in dataset after matching
        (scanner time stamps) and post-processing (LV locomotion data).
        """
        for i, fpath_nik in enumerate(fpaths_nik_time_stamps):
            nik_time_stamps = ND2TimeStamps(fpath_nik, "utf-16", "\t")
            lv_time_stamps = LabViewTimeStamps(fpaths_lv_time_stamps[i], "utf-8", "\t")
            lv_data = LabViewData(fpaths_lv_data[i], "utf-8", "\t")

            linear_locomotion = LinearLocomotion(
                nik_time_stamps, lv_time_stamps, lv_data, break_after_matching=True
            )
            # test scanner time stamps: these are not affected by post-processing
            assert isinstance(linear_locomotion.scanner_time_stamps, pd.Series)
            assert np.isclose(
                linear_locomotion.scanner_time_stamps_ms,
                expected_tsscn_after_match[i]["tsscn"],  # dataframe with one column [0]
                atol=0.0001,
            ).all()
            # compare labview data with expected after matching
            # TODO: add tests for lv_data vs expected!
            df_expected = expected_belt_after_match[i]
            for col in df_expected.columns:
                assert series_equal(linear_locomotion.lv_data[col], df_expected[col])
        # TODO: add tsscn data! include normal df and df_tsscn in same hdf5 file.

    @pytest.fixture(name="fpaths_expected_after_arduino_corr", scope="class")
    def fixture_fpaths_expected_after_arduino_corr(self, data_fpaths):
        """
        The file path of the expected data after the arduino artifact correction

        Args:
            data_folder (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [data_fpath[8] for data_fpath in data_fpaths]

    @pytest.fixture(name="expected_after_arduino_corr", scope="class")
    def fixture_expected_after_arduino_corr(
        self,
        fpaths_expected_after_arduino_corr,
    ) -> Tuple[pd.DataFrame]:
        """
        The expected data after the arduino artifact correction

        Args:
            fpath_expected_time_stamps (_type_): _description_

        Returns:
            Tuple[List[pd.DataFrame]]:
            Two lists as a tuple: a list of dataframes with original (labview) time stamps and
                all recorded quantities (speed, distance, rounds, stripes, ...),
                and a list of dataframes with the scanner time stamps.
        """
        dfs_list = []
        dfs_tsscn_list = []
        for fpath in fpaths_expected_after_arduino_corr:
            df = pd.read_hdf(fpath, key="df")
            df_tsscn = pd.read_hdf(fpath, key="df_tsscn")
            df.rename(columns=DICT_MATLAB_PYTHON_VARIABLES, inplace=True)
            # check if time_total_s is in milliseconds; if so, convert to seconds
            # assume never image < 1 Hz and there are at most 20 rounds.
            if (
                np.mean(
                    np.sort(np.diff(df["time_total_s"]))[
                        : int(0.05 * len(df["time_total_s"]))
                    ]
                )
                > 1
            ):
                df["time_total_s"] = df["time_total_s"] / 1000.0
            if (  # need abs because x -> 0 jump happens at new round for time per round
                np.abs(
                    np.mean(
                        np.sort(np.diff(df["time_per_round"]))[
                            : int(0.05 * len(df["time_per_round"]))
                        ]
                    )
                )
                > 1
            ):
                df["time_per_round"] = df["time_per_round"] / 1000.0

            dfs_list.append(df)
            dfs_tsscn_list.append(df_tsscn)
        return (dfs_list, dfs_tsscn_list)

    def test_after_arduino_corr(
        self,
        fpaths_nik_time_stamps,
        fpaths_lv_time_stamps,
        fpaths_lv_data,
        expected_after_arduino_corr,
    ):
        """
        Test the LinearLocomotion class with all recordings in dataset until step of
        arduino artifact correction.
        """
        dfs_expected, dfs_tsscn_expected = expected_after_arduino_corr
        for i, fpath_nik in enumerate(fpaths_nik_time_stamps):
            nik_time_stamps = ND2TimeStamps(fpath_nik, "utf-16", "\t")
            lv_time_stamps = LabViewTimeStamps(fpaths_lv_time_stamps[i], "utf-8", "\t")
            lv_data = LabViewData(fpaths_lv_data[i], "utf-8", "\t")

            linear_locomotion = LinearLocomotion(
                nik_time_stamps, lv_time_stamps, lv_data, break_after_arduino_corr=True
            )
            df_expected = dfs_expected[i]
            for col in df_expected.columns:
                assert series_equal(linear_locomotion.lv_data[col], df_expected[col])
            # TODO: add test tsscn

    @pytest.fixture(name="fpaths_expected_after_belt_corr", scope="class")
    def fixture_fpaths_expected_after_belt_corr(self, data_fpaths):
        """
        The file path of the expected data after the belt length correction

        Args:
            data_folder (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [data_fpath[9] for data_fpath in data_fpaths]

    @pytest.fixture(name="expected_after_belt_corr", scope="class")
    def fixture_expected_after_belt_corr(
        self,
        fpaths_expected_after_belt_corr,
    ) -> Tuple[pd.DataFrame]:
        """
        The expected data after the arduino artifact correction

        Args:
            fpath_expected_time_stamps (_type_): _description_

        Returns:
            Tuple[List[pd.DataFrame]]:
            Two lists as a tuple: a list of dataframes with original (labview) time stamps and
                all recorded quantities (speed, distance, rounds, stripes, ...),
                and a list of dataframes with the scanner time stamps.
        """
        dfs_list = []
        dfs_tsscn_list = []
        for fpath in fpaths_expected_after_belt_corr:
            df = pd.read_hdf(fpath, key="df")
            df_tsscn = pd.read_hdf(fpath, key="df_tsscn")
            df.rename(columns=DICT_MATLAB_PYTHON_VARIABLES, inplace=True)
            # check if time_total_s is in milliseconds; if so, convert to seconds
            # assume never image < 1 Hz and there are at most 20 rounds.
            if (
                np.mean(
                    np.sort(np.diff(df["time_total_s"]))[
                        : int(0.05 * len(df["time_total_s"]))
                    ]
                )
                > 1
            ):
                df["time_total_s"] = df["time_total_s"] / 1000.0
            if (  # need abs because x -> 0 jump happens at new round for time per round
                np.abs(
                    np.mean(
                        np.sort(np.diff(df["time_per_round"]))[
                            : int(0.05 * len(df["time_per_round"]))
                        ]
                    )
                )
                > 1
            ):
                df["time_per_round"] = df["time_per_round"] / 1000.0

            dfs_list.append(df)
            dfs_tsscn_list.append(df_tsscn)
        return (dfs_list, dfs_tsscn_list)

    def test_after_belt_corr(
        self,
        fpaths_nik_time_stamps,
        fpaths_lv_time_stamps,
        fpaths_lv_data,
        expected_after_belt_corr,
    ):
        """
        Test the LinearLocomotion class with all recordings in dataset until step of
        belt length correction.
        """
        dfs_expected, dfs_tsscn_expected = expected_after_belt_corr
        for i, fpath_nik in enumerate(fpaths_nik_time_stamps):
            nik_time_stamps = ND2TimeStamps(fpath_nik, "utf-16", "\t")
            lv_time_stamps = LabViewTimeStamps(fpaths_lv_time_stamps[i], "utf-8", "\t")
            lv_data = LabViewData(fpaths_lv_data[i], "utf-8", "\t")

            linear_locomotion = LinearLocomotion(
                nik_time_stamps, lv_time_stamps, lv_data, break_after_belt_corr=True
            )
            df_expected = dfs_expected[i]
            for col in df_expected.columns:
                assert series_equal(linear_locomotion.lv_data[col], df_expected[col])

    @pytest.fixture(name="fpaths_expected_after_pipeline", scope="class")
    def fixture_fpaths_expected_after_pipeline(self, data_fpaths):
        """
        The file path of the expected data after the pipeline

        Args:
            data_folder (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [data_fpath[10] for data_fpath in data_fpaths]

    @pytest.fixture(name="expected_after_pipeline", scope="class")
    def fixture_expected_after_pipeline(self, fpaths_expected_after_pipeline):
        """
        The expected data after the pipeline

        Args:
            fpath_expected_time_stamps (_type_): _description_

        Returns:
            Tuple[List[pd.DataFrame]]:
            Two lists as a tuple: a list of dataframes with original (labview) time stamps and
                all recorded quantities (speed, distance, rounds, stripes, ...),
                and a list of dataframes with the scanner time stamps.
        """
        dfs_list = []
        dfs_tsscn_list = []
        for fpath in fpaths_expected_after_pipeline:
            df = pd.read_hdf(fpath, key="df")
            df_tsscn = pd.read_hdf(fpath, key="df_tsscn")
            df.rename(columns=DICT_MATLAB_PYTHON_VARIABLES, inplace=True)
            # check if time_total_s is in milliseconds; if so, convert to seconds
            # assume never image < 1 Hz and there are at most 20 rounds.
            if (
                np.mean(
                    np.sort(np.diff(df["time_total_s"]))[
                        : int(0.05 * len(df["time_total_s"]))
                    ]
                )
                > 1
            ):
                df["time_total_s"] = df["time_total_s"] / 1000.0
            if (  # need abs because x -> 0 jump happens at new round for time per round
                np.abs(
                    np.mean(
                        np.sort(np.diff(df["time_per_round"]))[
                            : int(0.05 * len(df["time_per_round"]))
                        ]
                    )
                )
                > 1
            ):
                df["time_per_round"] = df["time_per_round"] / 1000.0

            dfs_list.append(df)
            dfs_tsscn_list.append(df_tsscn)
        return (dfs_list, dfs_tsscn_list)

    def test_after_pipeline(
        self,
        fpaths_nik_time_stamps,
        fpaths_lv_time_stamps,
        fpaths_lv_data,
        expected_after_pipeline,
    ):
        """
        Test the LinearLocomotion class with all recordings in dataset until step of
        belt length correction.
        """
        dfs_expected, dfs_tsscn_expected = expected_after_pipeline
        for i, fpath_nik in enumerate(fpaths_nik_time_stamps):
            nik_time_stamps = ND2TimeStamps(fpath_nik, "utf-16", "\t")
            lv_time_stamps = LabViewTimeStamps(fpaths_lv_time_stamps[i], "utf-8", "\t")
            lv_data = LabViewData(fpaths_lv_data[i], "utf-8", "\t")

            linear_locomotion = LinearLocomotion(
                nik_time_stamps, lv_time_stamps, lv_data
            )
            df_expected = dfs_expected[i]
            # check everything except "speed" and "running" (the only two that should have changed)
            for col in df_expected.columns:
                if col == "runtime" or col == "tsscn" or col == "speed" or col == "running":  # TODO: add runtime?
                    continue
                assert series_equal(linear_locomotion.lv_data[col], df_expected[col])
            # check running
            assert series_equal(linear_locomotion.lv_data["running"], df_expected["running"])
            # check speed; matlab units: mm/ms, python units: mm/s
            assert np.isclose(linear_locomotion.lv_data["speed"], df_expected["speed"]*1000.).all()
