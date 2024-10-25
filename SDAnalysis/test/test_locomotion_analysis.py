import pytest
import pandas as pd
import os
import sys
try:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, root_dir)
finally:
    from env_reader import read_env


@pytest.fixture(scope="module")
def data_folder():
    env_dict = read_env()
    return os.path.join(env_dict["TEST_DATA_FOLDER"], "Locomotion_analysis")


@pytest.fixture(scope="module")
def loco_chr2_fpath(data_folder):
    return os.path.join(data_folder, "loco_chr2_output.xlsx")


@pytest.fixture(scope="module")
def loco_chr2_delta_fpath(data_folder):
    return os.path.join(data_folder, "loco_chr2_delta_output.xlsx")


@pytest.fixture(scope="module")
def loco_aggregate_chr2_fpath(data_folder):
    return os.path.join(data_folder, "loco_chr2_aggregate_output.xlsx")


@pytest.fixture(scope="module")
def loco_aggregate_delta_chr2_fpath(data_folder):
    return os.path.join(data_folder, "loco_chr2_aggregate_delta_output.xlsx")


def test_data_folder_exists(data_folder):
    assert os.path.exists(data_folder)


def test_loco_chr2_files_exist(loco_chr2_fpath, loco_chr2_delta_fpath, loco_aggregate_chr2_fpath, loco_aggregate_delta_chr2_fpath):
    assert os.path.exists(loco_chr2_fpath)
    assert os.path.exists(loco_chr2_delta_fpath)
    assert os.path.exists(loco_aggregate_chr2_fpath)
    assert os.path.exists(loco_aggregate_delta_chr2_fpath)


@pytest.fixture(scope="module")
def loco_chr2_df(loco_chr2_fpath):
    return pd.read_excel(loco_chr2_fpath)


@pytest.fixture(scope="module")
def loco_chr2_delta_df(loco_chr2_delta_fpath):
    return pd.read_excel(loco_chr2_delta_fpath)


@pytest.fixture(scope="module")
def loco_aggregate_chr2(loco_aggregate_chr2_fpath):
    return pd.read_excel(loco_aggregate_chr2_fpath)


@pytest.fixture(scope="module")
def loco_aggregate_delta_chr2(loco_aggregate_delta_chr2_fpath):
    return pd.read_excel(loco_aggregate_delta_chr2_fpath)
