import os
import pandas as pd

for root, folders, files in os.walk("."):
    for file in files:
        if "expected_after_arduino_corr" in file:  # "expected_after_belt_corr", or "expected_after_pipeline"
            print(file)
            fpath = os.path.join(root, file)
            dfs = pd.read_excel(fpath, header=None, sheet_name=None)
            df_tsscn = dfs.pop("tsscn")
            df_rest = pd.concat(
                [
                    dfs[sheet_name].rename(columns={0: sheet_name})
                    for sheet_name in dfs.keys()
                ],
                axis=1,
            )
            # fpath_tsscn = os.path.splitext(fpath)[0] + "_tsscn.hdf5"
            fpath_rest = os.path.splitext(fpath)[0] + ".hdf5"
            df_rest.to_hdf(fpath_rest, index=False, key="df")
            df_tsscn.to_hdf(fpath_rest, index=False, key="df_tsscn")
