import os
import sys
import pandas as pd

try:
    project_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )  # SDAnalysis/SDAnalysis folder
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )  # SDAnalysis folder (top-level folder)
    sys.path.insert(0, root_dir)
    sys.path.insert(0, project_dir)

except:
    raise Exception("Exception while adding root_dir to sys.path")
finally:
    from sdanalysis.metadata_reader import MetadataReader as MR

def test_metadata_folder_exists():
    """Test that the METADATA_FOLDER exists."""
    env_dict = dict()
    fpath_env = os.path.join(root_dir, ".env")
    assert os.path.exists(fpath_env)
    if not os.path.exists(fpath_env):
        print(".env does not exist")
    else:
        with open(fpath_env, "r") as f:
            for line in f.readlines():
                l = line.rstrip().split("=")
                env_dict[l[0]] = l[1]
    assert len(env_dict) > 0
    assert "METADATA_FOLDER" in env_dict
    assert os.path.exists(env_dict["METADATA_FOLDER"])

def test_metadata_duckdb_exists():
    """Test that the metadata.duckdb file exists."""
    env_dict = dict()
    fpath_env = os.path.join(root_dir, ".env")
    assert os.path.exists(fpath_env)
    if not os.path.exists(fpath_env):
        print(".env does not exist")
    else:
        with open(fpath_env, "r") as f:
            for line in f.readlines():
                l = line.rstrip().split("=")
                env_dict[l[0]] = l[1]
    assert len(env_dict) > 0
    assert "METADATA_FOLDER" in env_dict
    fpath_duckdocu = os.path.join(
        env_dict["METADATA_FOLDER"], "metadata.duckdb"
    )
    assert os.path.exists(fpath_duckdocu)

def test_metadata_reader_consistent_alternatives():
    """Test that opening duckdb file vs opening folder with metadata yields the same result."""
    env_dict = dict()
    fpath_env = os.path.join(root_dir, ".env")
    assert os.path.exists(fpath_env)
    if not os.path.exists(fpath_env):
        print(".env does not exist")
    else:
        with open(fpath_env, "r") as f:
            for line in f.readlines():
                l = line.rstrip().split("=")
                env_dict[l[0]] = l[1]
    assert len(env_dict) > 0
    assert "METADATA_FOLDER" in env_dict
    # load from folder
    mdata1 = MR(env_dict["METADATA_FOLDER"])
    mdata1._load_metadata()
    # load from duckdb
    fpath_duckdocu = os.path.join(
        env_dict["METADATA_FOLDER"], "metadata.duckdb"
    )
    assert os.path.exists(fpath_duckdocu)
    mdata2 = MR(fpath_duckdocu)
    mdata2._load_metadata()
    # check if the two dataframes are the same
    # index might differ, so ignore it
    df1 = mdata1.grouping_df.reset_index().sort_values(by="uuid").drop("index", axis=1)
    df2 = mdata2.grouping_df.reset_index().sort_values(by="uuid").drop("index", axis=1)
    assert _dfs_equal(df1, df2)

    df1 = (
        mdata1.annotation_df.sort_values(by="nd2").reset_index().drop("index", axis=1)
    )
    df2 = (
        mdata2.annotation_df.sort_values(by="nd2").reset_index().drop("index", axis=1)
    )
    assert _dfs_equal(df1, df2)

    df1 = (
        mdata1.colorings_df.reset_index()
        .sort_values(by="mouse_id")
        .drop("index", axis=1)
    )
    df2 = (
        mdata2.colorings_df.reset_index()
        .sort_values(by="mouse_id")
        .drop("index", axis=1)
    )
    assert _dfs_equal(df1, df2)

    df1 = (
        mdata1.win_inj_types_df.reset_index()
        .sort_values(by="mouse_id")
        .drop("index", axis=1)
    )
    df2 = (
        mdata2.win_inj_types_df.reset_index()
        .sort_values(by="mouse_id")
        .drop("index", axis=1)
    )
    assert _dfs_equal(df1, df2)

    df1 = (
        mdata1.events_df.reset_index()
        .sort_values(by=["event_uuid", "event_index"])
        .drop("index", axis=1)
    )
    df2 = (
        mdata2.events_df.reset_index()
        .sort_values(by=["event_uuid", "event_index"])
        .drop("index", axis=1)
    )
    assert _dfs_equal(df1, df2)


def _dfs_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    row_comparisons = df1 == df2
    # correct np.NaN != np.NaN artifact
    row_comparisons[pd.isnull(df1) & pd.isnull(df2)] = True
    # ignore "folder" in groupings, as server folders were removed for publication. Make sure to check presence of a few columns to assure it is the grouping_df.
    if "folder" in row_comparisons.columns and "nd2" in row_comparisons.columns and "day" in row_comparisons.columns:
        row_comparisons.folder = True
    # first all() is aggregation over rows, second is over columns
    return row_comparisons.all().all()
