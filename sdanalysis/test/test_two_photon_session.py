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
# TODO: make a very small dataset: LFP, labview and nd2 with both only green and green + red.
# TODO: make it a proper test file in the future (pytest). Need small data first.

nd2_green_fname = "T386_20211202_green.nd2"
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
    env_dict = read_env()
    # Test data for TPS is in test data folder -> Test2pSession
    return os.path.join(env_dict["TEST_DATA_FOLDER"], "Test_2p_session")


@pytest.fixture(name="matlab_2p_folder", scope="module")
def fixture_matlab_2p_folder():
    env_dict = read_env()
    return env_dict["MATLAB_2P_FOLDER"]


@pytest.fixture(name="session_1ch_fpaths", scope="module")
def fixture_session_1ch_fpaths(data_folder):
    return [
        os.path.join(data_folder, nd2_green_fname),
        os.path.join(data_folder, ND2_GREEN_NIK),
        os.path.join(data_folder, ND2_GREEN_LV),
        os.path.join(data_folder, ND2_GREEN_LVTIME),
        os.path.join(data_folder, ND2_GREEN_LFP),
    ]


@pytest.fixture(name="session_2ch_fpaths", scope="module")
def fixture_session_2ch_fpaths(data_folder):
    return [
        os.path.join(data_folder, ND2_DUAL_FNAME),
        os.path.join(data_folder, ND2_DUAL_NIK),
        os.path.join(data_folder, ND2_DUAL_LV),
        os.path.join(data_folder, ND2_DUAL_LVTIME),
        os.path.join(data_folder, ND2_DUAL_LFP),
    ]


class TestFilesExist:
    def test_1ch_files_found(self, session_1ch_fpaths):
        for fpath in session_1ch_fpaths:
            assert os.path.exists(fpath)

    def test_1ch_output_exists(self, session_1ch_output_fpath):
        assert os.path.exists(session_1ch_output_fpath)

    def test_2ch_files_found(self, session_2ch_fpaths):
        for fpath in session_2ch_fpaths:
            assert os.path.exists(fpath)

    def test_2ch_output_exists(self, session_2ch_output_fpath):
        assert os.path.exists(session_2ch_output_fpath)


# TODO: full/not full TPS files should be tested both?
# TODO: test other scenarios (one source missing: LFP, LV or Nik)


@pytest.fixture(name="session_1ch_output_fpath", scope="module")
def fixture_session_1ch_output_fpath(data_folder):
    return os.path.join(os.path.join(data_folder, "tps"), "tps_green_notfull.hdf5")


@pytest.fixture(name="session_2ch_output_fpath", scope="module")
def fixture_session_2ch_output_fpath(data_folder):
    return os.path.join(os.path.join(data_folder, "tps"), "tps_dual_notfull.hdf5")


def test_data_folder_exists(data_folder):
    assert os.path.exists(data_folder)


def test_matlab_2p_folder_exists(matlab_2p_folder):
    assert os.path.exists(matlab_2p_folder)


def _check_tps_hdf5_structure(hf: h5py.File):
    assert hf is not None
    # should be ('basic', '/basic'), ('inferred', '/inferred'), ('mean_fluo', 'mean_fluo')
    its = hf.items()
    assert len(its) == 3
    assert sorted([it[0] for it in hf.items()]) == ["basic", "inferred", "mean_fluo"]
    # test content structure of basic group
    grp_basic = hf["basic"]
    assert sorted([it[0] for it in grp_basic.items()]) == [
        "LABVIEW_PATH",
        "LABVIEW_TIMESTAMPS_PATH",
        "LFP_PATH",
        "MATLAB_2P_FOLDER",
        "ND2_PATH",
        "ND2_TIMESTAMPS_PATH",
    ]
    # test inferred group
    grp_inferred = hf["inferred"]
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
    bools = arr1 == arr2
    bools[np.isnan(arr1) & np.isnan(arr2)] = True
    assert np.all(bools)


def _compare_sessions(ses1, ses2):
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
    # check inferred group
    for k in ses1.belt_dict.keys():  # only arrays in belt_dict
        assert k in ses2.belt_dict
        _compare_arrays(ses1.belt_dict[k], ses2.belt_dict[k])
    for k in ses1.belt_scn_dict.keys():  # only arrays in belt_scn_dict
        assert k in ses2.belt_scn_dict
        _compare_arrays(ses1.belt_scn_dict[k], ses2.belt_scn_dict[k])
    for k in ses1.belt_params.keys():
        assert k in ses2.belt_params
        v1 = ses1.belt_params[k]
        v2 = ses2.belt_params[k]
        if isinstance(v1, np.ndarray):
            _compare_arrays(v1, v2)
        elif k == "path_name":  # a string path (path_name)
            assert os.path.split(v1)[1] == os.path.split(v2)[1]
        else:
            assert v1 == v2
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
        assert isinstance(session_1ch, tps.TwoPhotonSession)
        assert isinstance(session_1ch_loaded, tps.TwoPhotonSession)

    def test_1ch_output_structure(self, session_1ch_output_fpath):
        with h5py.File(session_1ch_output_fpath, "r") as hf:
            _check_tps_hdf5_structure(hf)

    def test_1ch_output_open(self, session_1ch_loaded):
        assert isinstance(session_1ch_loaded, tps.TwoPhotonSession)
        # assert session_1ch_loaded.LABVIEW_PATH is not None

    def test_tps_1ch_results(self, session_1ch, session_1ch_loaded):
        _compare_sessions(session_1ch, session_1ch_loaded)
        # FIXME: first session (session_1ch) is all none
        # The problem seems to be in the init_and_process method:
        # belt_dict is None, so nothing gets copied...
        # So maybe the instance._load_preprocess_data() function in the first step is bad


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
        assert isinstance(session_2ch, tps.TwoPhotonSession)
        assert isinstance(session_2ch_loaded, tps.TwoPhotonSession)

    def test_2ch_output_structure(self, session_2ch_output_fpath):
        with h5py.File(session_2ch_output_fpath, "r") as hf:
            _check_tps_hdf5_structure(hf)

    def test_2ch_output_open(self, session_2ch_loaded):
        assert isinstance(session_2ch_loaded, tps.TwoPhotonSession)

    def test_tps_2ch_results(self, session_2ch, session_2ch_loaded):
        _compare_sessions(session_2ch, session_2ch_loaded)


# TODO: add tests to reading into a twophotonsession from hdf5 (make a small dataset)
