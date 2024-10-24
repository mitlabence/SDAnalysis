import sys
import os
import pytest
try:  # need to keep order: path.insert, then import.
    # Get the absolute path of the parent directory (the root folder)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Add the libs folder to the system path
    sys.path.insert(0, root_dir)
finally:
    import two_photon_session as tps
    from env_reader import read_env
# TODO: make a very small dataset: LFP, labview and nd2 with both only green and green + red.
# TODO: make it a proper test file in the future (pytest). Need small data first.

FOLDER = os.path.normpath("D:\PhD\Data\T386_MatlabTest")

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


@pytest.fixture
def data_folder():
    env_dict = read_env()
    return env_dict["TEST_DATA_FOLDER"]


@pytest.fixture
def matlab_2p_folder():
    env_dict = read_env()
    return env_dict["MATLAB_2P_FOLDER"]


@pytest.fixture
def session_1ch_fpaths(data_folder):
    return [
        os.path.join(data_folder, ND2_GREEN_FNAME),
        os.path.join(data_folder, ND2_GREEN_NIK),
        os.path.join(data_folder, ND2_GREEN_LV),
        os.path.join(data_folder, ND2_GREEN_LVTIME),
        os.path.join(data_folder, ND2_GREEN_LFP)
    ]


@pytest.fixture
def session_2ch_fpaths(data_folder):
    return [
        os.path.join(data_folder, ND2_DUAL_FNAME),
        os.path.join(data_folder, ND2_DUAL_NIK),
        os.path.join(data_folder, ND2_DUAL_LV),
        os.path.join(data_folder, ND2_DUAL_LVTIME),
        os.path.join(data_folder, ND2_DUAL_LFP)
    ]


def test_data_folder_exists(data_folder):
    assert os.path.exists(data_folder)


def test_matlab_2p_folder_exists(matlab_2p_folder):
    assert os.path.exists(matlab_2p_folder)


def test_1ch_files_found(session_1ch_fpaths):
    for fpath in session_1ch_fpaths:
        assert os.path.exists(fpath)


def test_2ch_files_found(session_2ch_fpaths):
    for fpath in session_2ch_fpaths:
        assert os.path.exists(fpath)


def test_tps_open_1ch_files(session_1ch_fpaths, matlab_2p_folder):
    session = tps.TwoPhotonSession.init_and_process(
        *session_1ch_fpaths, matlab_2p_folder=matlab_2p_folder)
    assert session is not None
    # TODO: add proper assertions regarding output


def test_tps_open_2ch_files(session_2ch_fpaths, matlab_2p_folder):
    session = tps.TwoPhotonSession.init_and_process(
        *session_2ch_fpaths, matlab_2p_folder=matlab_2p_folder)
    assert session is not None
    # TODO: add proper assertions!
