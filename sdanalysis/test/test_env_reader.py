import sys
import os
import warnings
import pytest


try:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root_dir)
finally:
    from env_reader import read_env


@pytest.fixture(scope="module")
def env_dict():
    env_dict = read_env()
    return env_dict


def test_env_reader_success(env_dict):
    assert isinstance(env_dict, dict)
    assert len(env_dict) > 0


def test_env_reader_different_paths():
    os.chdir("..")
    env_dict = read_env()
    assert len(env_dict) > 0
    assert "TEST_DATA_FOLDER" in env_dict

    os.chdir("..")
    env_dict = read_env()
    assert len(env_dict) > 0
    assert "TEST_DATA_FOLDER" in env_dict


def test_env_dict_contents(env_dict):
    """Test the actual contents of the env_dict

    Args:
        env_dict (pyfixture variable): the dictionary from the .env
    """
    assert "DOWNLOADS_FOLDER" in env_dict
    assert "TEST_DATA_FOLDER" in env_dict
    assert "OUTPUT_FOLDER" in env_dict
    assert "DATA_DOCU_FOLDER" in env_dict
    assert "LOG_FOLDER" in env_dict
    if not "SERVER_SYMBOL" in env_dict:
        warnings.warn("SERVER_SYMBOL not found in env_dict")
    for k, fpath in env_dict.items():
        if k == "MATLAB_2P_FOLDER":
            continue  # matlab-2p is deprecated
        assert os.path.exists(fpath)
