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
    from data_documentation import DataDocumentation as DD


def test_data_documentation_consistent_alternatives():
    """Test that opening duckdb file vs opening folder with data documentation yields the same result."""
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
    assert "DATA_DOCU_FOLDER" in env_dict
    # load from folder
    ddoc1 = DD(env_dict["DATA_DOCU_FOLDER"])
    ddoc1.loadDataDoc()
    # load from duckdb
    fpath_duckdocu = os.path.join(
        env_dict["DATA_DOCU_FOLDER"], "data_documentation.duckdb"
    )
    assert os.path.exists(fpath_duckdocu)
    ddoc2 = DD(fpath_duckdocu)
    ddoc2.loadDataDoc()
    # check if the two dataframes are the same
    # index might differ, so ignore it
    df1 = ddoc1.grouping_df.reset_index().sort_values(by="uuid").drop("index", axis=1)
    df2 = ddoc2.grouping_df.reset_index().sort_values(by="uuid").drop("index", axis=1)
    assert _dfs_equal(df1, df2)

    df1 = (
        ddoc1.segmentation_df.reset_index().sort_values(by="nd2").drop("index", axis=1)
    )
    df2 = (
        ddoc2.segmentation_df.reset_index().sort_values(by="nd2").drop("index", axis=1)
    )
    assert _dfs_equal(df1, df2)

    df1 = (
        ddoc1.colorings_df.reset_index()
        .sort_values(by="mouse_id")
        .drop("index", axis=1)
    )
    df2 = (
        ddoc2.colorings_df.reset_index()
        .sort_values(by="mouse_id")
        .drop("index", axis=1)
    )
    assert _dfs_equal(df1, df2)

    df1 = (
        ddoc1.win_inj_types_df.reset_index()
        .sort_values(by="mouse_id")
        .drop("index", axis=1)
    )
    df2 = (
        ddoc2.win_inj_types_df.reset_index()
        .sort_values(by="mouse_id")
        .drop("index", axis=1)
    )
    assert _dfs_equal(df1, df2)

    df1 = (
        ddoc1.events_df.reset_index()
        .sort_values(by=["event_uuid", "event_index"])
        .drop("index", axis=1)
    )
    df2 = (
        ddoc2.events_df.reset_index()
        .sort_values(by=["event_uuid", "event_index"])
        .drop("index", axis=1)
    )
    assert _dfs_equal(df1, df2)


def _dfs_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    row_comparisons = df1 == df2
    # correct np.NaN != np.NaN artifact
    row_comparisons[pd.isnull(df1) & pd.isnull(df2)] = True
    # first all() is aggregation over rows, second is over columns
    return row_comparisons.all().all()
