import sys
import os
import pytest
import warnings

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
    assert "DOWNLOADS_FOLDER" in env_dict
    assert "TEST_DATA_FOLDER" in env_dict
    assert "OUTPUT_FOLDER" in env_dict
    assert "DATA_DOCU_FOLDER" in env_dict
    assert "LOG_FOLDER" in env_dict
    assert "MATLAB_2P_FOLDER" in env_dict
    if not "SERVER_SYMBOL" in env_dict:
        warnings.warn("SERVER_SYMBOL not found in env_dict")
