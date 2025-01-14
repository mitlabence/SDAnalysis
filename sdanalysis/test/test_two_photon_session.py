"""
test_two_photon_session.py - Test the TwoPhotonSession class.
"""

import sys
import os
import pytest
import h5py
import numpy as np

try:  # need to keep order: path.insert, then import.
    # Get the absolute path of the parent directory (the root folder)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Add the libs folder to the system path
    sys.path.insert(0, root_dir)
finally:
    import two_photon_session as tps
    from env_reader import read_env
    import constants

ND2_GREEN_FNAME = "T386_20211202_green.nd2"
ND2_GREEN_LFP = "21d02000.abf"
ND2_GREEN_LV = "T386.021221.1105.txt"
ND2_GREEN_LVTIME = "T386.021221.1105time.txt"
ND2_GREEN_NIK = "T386.021221.1105_nik.txt"

ND2_DUAL_FNAME = "T386_20211202_green_red.nd2"
ND2_DUAL_LFP = "21d02001.abf"
ND2_DUAL_LV = "T386.021221.1106.txt"
ND2_DUAL_LVTIME = "T386.021221.1106time.txt"
ND2_DUAL_NIK = "T386.021221.1106_nik.txt"


@pytest.fixture(name="data_folder", scope="module")
def fixture_data_folder():
    """
    The data folder for the test data. This is the folder where the test data is stored.

    Returns:
        _type_: _description_
    """
    env_dict = read_env()
    # Test data for TPS is in test data folder -> Test2pSession
    return os.path.join(env_dict["TEST_DATA_FOLDER"], "Test_2p_session")


@pytest.fixture(name="matlab_2p_folder", scope="module")
def fixture_matlab_2p_folder():
    """
    The folder where the matlab-2p code is stored.

    Returns:
        _type_: _description_
    """
    env_dict = read_env()
    return env_dict["MATLAB_2P_FOLDER"]


@pytest.fixture(name="session_1ch_fpaths", scope="module")
def fixture_session_1ch_fpaths(data_folder):
    """
    File paths of the single-channel test data.

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    return [
        os.path.join(data_folder, ND2_GREEN_FNAME),
        os.path.join(data_folder, ND2_GREEN_NIK),
        os.path.join(data_folder, ND2_GREEN_LV),
        os.path.join(data_folder, ND2_GREEN_LVTIME),
        os.path.join(data_folder, ND2_GREEN_LFP),
    ]


@pytest.fixture(name="session_2ch_fpaths", scope="module")
def fixture_session_2ch_fpaths(data_folder):
    """
    File paths of the dual-channel test data.

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    return [
        os.path.join(data_folder, ND2_DUAL_FNAME),
        os.path.join(data_folder, ND2_DUAL_NIK),
        os.path.join(data_folder, ND2_DUAL_LV),
        os.path.join(data_folder, ND2_DUAL_LVTIME),
        os.path.join(data_folder, ND2_DUAL_LFP),
    ]


@pytest.fixture(name="session_1ch_output_fpath", scope="module")
def fixture_session_1ch_output_fpath(data_folder):
    """
    The expected output file for the single-channel test data.

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    return os.path.join(os.path.join(data_folder, "tps"), "tps_green_notfull.hdf5")


@pytest.fixture(name="session_2ch_output_fpath", scope="module")
def fixture_session_2ch_output_fpath(data_folder):
    """
    The expected output file for the dual-channel test data.

    Args:
        data_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    return os.path.join(os.path.join(data_folder, "tps"), "tps_dual_notfull.hdf5")


class TestFilesExist:
    """Class grouping tests for the existence of the test files"""

    @staticmethod
    def test_data_folder_exists(data_folder):
        """
        Test if the data folder exists.

        Args:
            data_folder (_type_): _description_
        """
        assert os.path.exists(data_folder)

    def test_1ch_files_found(self, session_1ch_fpaths):
        """
        Test if the single-channel test files exist.

        Args:
            session_1ch_fpaths (_type_): _description_
        """
        for fpath in session_1ch_fpaths:
            assert os.path.exists(fpath)

    def test_1ch_output_exists(self, session_1ch_output_fpath):
        """
        Test if the single-channel expected output file exists.

        Args:
            session_1ch_output_fpath (_type_): _description_
        """
        assert os.path.exists(session_1ch_output_fpath)

    def test_2ch_files_found(self, session_2ch_fpaths):
        """
        Test if the dual-channel test files exist.

        Args:
            session_2ch_fpaths (_type_): _description_
        """
        for fpath in session_2ch_fpaths:
            assert os.path.exists(fpath)

    def test_2ch_output_exists(self, session_2ch_output_fpath):
        """
        Test if the dual-channel expected output file exists.

        Args:
            session_2ch_output_fpath (_type_): _description_
        """
        assert os.path.exists(session_2ch_output_fpath)


# TODO: test other scenarios (one source missing: LFP, LV or Nik)


def _check_tps_hdf5_structure(hdf_file: h5py.File):
    """
    Check the structure of the TwoPhotonSession HDF5 file.

    Args:
        hf (h5py.File): _description_
    """
    assert hdf_file is not None
    # should be ('basic', '/basic'), ('inferred', '/inferred'), ('mean_fluo', 'mean_fluo')
    its = hdf_file.items()
    assert len(its) == 3
    assert sorted([it[0] for it in hdf_file.items()]) == [
        "basic",
        "inferred",
        "mean_fluo",
    ]
    # test content structure of basic group
    grp_basic = hdf_file["basic"]
    assert sorted([it[0] for it in grp_basic.items()]) == [
        "LABVIEW_PATH",
        "LABVIEW_TIMESTAMPS_PATH",
        "LFP_PATH",
        "MATLAB_2P_FOLDER",
        "ND2_PATH",
        "ND2_TIMESTAMPS_PATH",
    ]
    # test inferred group
    grp_inferred = hdf_file["inferred"]
    assert sorted([it[0] for it in grp_inferred.items()]) == [
        "belt_dict",
        "belt_params",
        "belt_scn_dict",
        "lfp_scaling",
        "lfp_t_start",
        "nik_t_start",
        "nikon_daq_time",
        "time_offs_lfp_nik",
    ]


def _compare_arrays(arr1, arr2):
    """
    Compare two numpy arrays. If the shape matches and all entries are the same, the arrays are
    considered equal. If an index location contains NaN in both arrays, they are considered equal.

    Args:
        arr1 (_type_): _description_
        arr2 (_type_): _description_
    """
    if np.issubdtype(arr1.dtype, np.floating) and np.issubdtype(
        arr2.dtype, np.floating
    ):
        assert np.allclose(arr1, arr2, equal_nan=True)
    else:
        assert np.array_equal(arr1, arr2, equal_nan=True)


def _compare_sessions(ses1, ses2):
    """
    Compare two TwoPhotonSession objects.

    Args:
        ses1 (_type_): _description_
        ses2 (_type_): _description_
    """
    # check basic group file names, except matlab-2p folder
    assert os.path.split(ses1.labview_path)[1] == os.path.split(ses2.labview_path)[1]
    assert (
        os.path.split(ses1.labview_timestamps_path)[1]
        == os.path.split(ses2.labview_timestamps_path)[1]
    )
    assert os.path.split(ses1.lfp_path)[1] == os.path.split(ses2.lfp_path)[1]
    assert os.path.split(ses1.nd2_path)[1] == os.path.split(ses2.nd2_path)[1]
    assert (
        os.path.split(ses1.nd2_timestamps_path)[1]
        == os.path.split(ses2.nd2_timestamps_path)[1]
    )
    is_matlab_1 = ses1.matlab_2p_folder is not None
    is_matlab_2 = ses2.matlab_2p_folder is not None
    # check inferred group
    for (
        _,
        k,
    ) in constants.DICT_MATLAB_PYTHON_VARIABLES.items():  # go through re-mapped keys
        assert k in ses1.belt_dict
        assert k in ses2.belt_dict
        if k == "time_per_round":
            _compare_arrays(ses1.belt_dict[k] * 1000, ses2.belt_dict[k])
        else:
            _compare_arrays(ses1.belt_dict[k], ses2.belt_dict[k])
    for (
        _,
        k,
    ) in (
        constants.DICT_MATLAB_PYTHON_SCN_VARIABLES.items()
    ):  # only arrays in belt_scn_dict
        assert k in ses1.belt_scn_dict
        assert k in ses2.belt_scn_dict
        if (
            k == "time_per_round"
        ):  # TODO: convert this to original ms, and add time_per_round_s?
            _compare_arrays(ses1.belt_scn_dict[k] * 1000, ses2.belt_scn_dict[k])
        else:
            _compare_arrays(ses1.belt_scn_dict[k], ses2.belt_scn_dict[k])
    for k in ses1.belt_params.keys():
        assert k in ses2.belt_params
        v_1 = ses1.belt_params[k]
        v_2 = ses2.belt_params[k]
        if isinstance(v_1, np.ndarray):
            _compare_arrays(v_1, v_2)
        elif k == "path_name":  # a string path (path_name)
            continue  # do not test the path, as it can differ
        elif "file" in k:
            len_shorter = min(len(v_1), len(v_2))
            assert (
                v_1[:len_shorter] == v_2[:len_shorter]
            )  # possible extensions can be missing
            # (in matlab)
        elif "i_belt" in k:  # i_belt_start and i_belt_stop: matlab has 1-based indexing
            if is_matlab_1:
                v_1 -= 1
            if is_matlab_2:
                v_2 -= 1
            assert v_1 == v_2
        elif isinstance(v_1, str):
            assert v_1.lower() == v_2.lower()
        elif "len_tsscn" in k:  # matlab did this one weirdly: in one example,
            # tsscn is 577 long in belt_scn_dict, but 578 in belt_dict,
            # it comes from cutting tsscn when creating belt_scn struct in matlab.
            continue
        elif k == "art_n_artifacts":  # not implemented (yet)
            continue
        elif k == "timestamps_were_duplicate":
            continue  # this is not used in the current code
        elif k == "movie_length_min":  # matlab has a bug, so do not compare
            continue
        elif k == "frequency_estimated":  # matlab has a bug, so do not compare
            continue
        elif k == "belt_length_mm":  # not implemented yet
            continue
        else:
            assert v_1 == v_2
    assert ses1.lfp_scaling == ses2.lfp_scaling
    assert ses1.lfp_t_start == ses2.lfp_t_start
    assert ses1.nik_t_start == ses2.nik_t_start
    _compare_arrays(ses1.nikon_daq_time, ses2.nikon_daq_time)
    assert ses1.time_offs_lfp_nik == ses2.time_offs_lfp_nik
    #  compare mean_fluo
    _compare_arrays(ses1.mean_fluo, ses2.mean_fluo)


@pytest.mark.usefixtures("session_1ch")
class TestTwoPhotonSession1Ch:
    """Class grouping tests for 1-channel imaging data"""

    @pytest.fixture(name="session_1ch_loaded", scope="class")
    def fixture_session_1ch_loaded(self, session_1ch_output_fpath):
        """The should-be output of TwoPhotonSession object (as a TPS object)"""
        return tps.TwoPhotonSession.from_hdf5(
            session_1ch_output_fpath, try_open_files=False
        )

    @pytest.fixture(name="session_1ch", scope="class")
    def fixture_session_1ch(self, session_1ch_fpaths, matlab_2p_folder):
        """The TwoPhotonSession object processing test files"""
        return tps.TwoPhotonSession.init_and_process(
            *session_1ch_fpaths, matlab_2p_folder=matlab_2p_folder
        )

    def test_1ch_setup_successful(self, session_1ch, session_1ch_loaded):
        """
        Test if the setup of the single-channel test is successful.

        Args:
            session_1ch (_type_): _description_
            session_1ch_loaded (_type_): _description_
        """
        assert isinstance(session_1ch, tps.TwoPhotonSession)
        assert isinstance(session_1ch_loaded, tps.TwoPhotonSession)

    def test_1ch_output_structure(self, session_1ch_output_fpath):
        """
        Test the structure of the single-channel output file.

        Args:
            session_1ch_output_fpath (_type_): _description_
        """
        with h5py.File(session_1ch_output_fpath, "r") as hdf_file:
            _check_tps_hdf5_structure(hdf_file)

    def test_1ch_output_open(self, session_1ch_loaded):
        """
        Test if the single-channel output file can be opened.

        Args:
            session_1ch_loaded (_type_): _description_
        """
        assert isinstance(session_1ch_loaded, tps.TwoPhotonSession)
        # assert session_1ch_loaded.LABVIEW_PATH is not None

    def test_tps_1ch_results(self, session_1ch, session_1ch_loaded):
        """
        Test the results of the single-channel TwoPhotonSession object.

        Args:
            session_1ch (_type_): _description_
            session_1ch_loaded (_type_): _description_
        """
        _compare_sessions(session_1ch, session_1ch_loaded)


@pytest.mark.usefixtures("session_2ch")
class TestTwoPhotonSession2Ch:
    """Class grouping tests for 2-channel imaging data"""

    @pytest.fixture(name="session_2ch", scope="class")
    def fixture_session_2ch(self, session_2ch_fpaths, matlab_2p_folder):
        """The TwoPhotonSession object processing test files"""
        return tps.TwoPhotonSession.init_and_process(
            *session_2ch_fpaths, matlab_2p_folder=matlab_2p_folder
        )

    @pytest.fixture(name="session_2ch_loaded", scope="class")
    def fixture_session_2ch_loaded(self, session_2ch_output_fpath):
        """The should-be output of TwoPhotonSession object (as a TPS object)"""
        return tps.TwoPhotonSession.from_hdf5(
            session_2ch_output_fpath, try_open_files=False
        )

    def test_2ch_setup_successful(self, session_2ch, session_2ch_loaded):
        """
        Test if the setup of the dual-channel test is successful.

        Args:
            session_2ch (_type_): _description_
            session_2ch_loaded (_type_): _description_
        """
        assert isinstance(session_2ch, tps.TwoPhotonSession)
        assert isinstance(session_2ch_loaded, tps.TwoPhotonSession)

    def test_2ch_output_structure(self, session_2ch_output_fpath):
        """
        Test the structure of the dual-channel output file.

        Args:
            session_2ch_output_fpath (_type_): _description_
        """
        with h5py.File(session_2ch_output_fpath, "r") as hdf_file:
            _check_tps_hdf5_structure(hdf_file)

    def test_2ch_output_open(self, session_2ch_loaded):
        """
        Test if the dual-channel output file can be opened.

        Args:
            session_2ch_loaded (_type_): _description_
        """
        assert isinstance(session_2ch_loaded, tps.TwoPhotonSession)

    def test_tps_2ch_results(self, session_2ch, session_2ch_loaded):
        """
        Test the results of the dual-channel TwoPhotonSession object.

        Args:
            session_2ch (_type_): _description_
            session_2ch_loaded (_type_): _description_
        """
        _compare_sessions(session_2ch, session_2ch_loaded)


# TODO: add tests to reading into a twophotonsession from hdf5 (make a small dataset)
