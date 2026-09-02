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


# test whether grouping_df "folder" is NaN for each row (published data should not have legacy folder paths)
def test_grouping_df_folder_is_nan():
    """Test that the grouping_df "folder" is NaN for each row."""
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
    fpath_duckdb = os.path.join(
        env_dict["METADATA_FOLDER"], "metadata.duckdb"
    )
    assert os.path.exists(fpath_duckdb)

    # read metadata from folders/files
    mdata = MR(fpath_duckdb)
    mdata._load_metadata()
    grouping_df = mdata.grouping_df
    assert isinstance(grouping_df, pd.DataFrame)
    assert "folder" in grouping_df.columns
    assert grouping_df["folder"].isna().all()
    print(grouping_df.experiment_type.unique())
    # read from duckdb file
    mdata2 = MR(fpath_duckdb)
    mdata2._load_metadata()
    grouping_df2 = mdata2.grouping_df
    assert isinstance(grouping_df2, pd.DataFrame)
    assert "folder" in grouping_df2.columns
    assert grouping_df2["folder"].isna().all()
