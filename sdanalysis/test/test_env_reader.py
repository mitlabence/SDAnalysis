import sys
import os
try:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, root_dir)
finally:
    from env_reader import read_env


def test_env_reader():
    env_dict = read_env()
    assert len(env_dict) > 0
    assert "TEST_DATA_FOLDER" in env_dict


def test_env_reader_different_paths():
    os.chdir("..")
    env_dict = read_env()
    assert len(env_dict) > 0
    assert "TEST_DATA_FOLDER" in env_dict

    os.chdir("..")
    env_dict = read_env()
    assert len(env_dict) > 0
    assert "TEST_DATA_FOLDER" in env_dict
