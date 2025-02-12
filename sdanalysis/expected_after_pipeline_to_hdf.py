import os
import pandas as pd

col_names = [
        "round",
        "speed",
        "total_distance",
        "distance_per_round",
        "reflectivity",
        "unknown",
        "stripes_total",
        "stripes_per_round",
        "time_total_ms",
        "time_per_round",
        "stimuli1",
        "stimuli2",
        "stimuli3",
        "stimuli4",
        "stimuli5",
        "stimuli6",
        "stimuli7",
        "stimuli8",
        "stimuli9",
        "pupil_area",
    ]

for root, folders, files in os.walk("."):
    for file in files:
        if "expected_after_pipeline_scn.xlsx" in file:
            continue  # handle this file once the other one is handled
        if "expected_after_pipeline.xlsx" in file:
            fpath = os.path.join(root, file)
            fpath_tsscn = os.path.join(root, os.path.splitext(file)[0] + "_scn.xlsx")
            print(fpath)
            assert os.path.exists(fpath_tsscn)
            fpath_out = os.path.splitext(fpath)[0] + ".hdf5"
            if os.path.exists(fpath_out):
                print(f"{fpath_out} already exists, skipping...")
                continue

            sheets = pd.read_excel(fpath, sheet_name=None, header=None)
            df = pd.concat([sheets[sheet].rename(columns={0: sheet}) for sheet in sheets], axis=1)

            sheets_tsscn = pd.read_excel(fpath_tsscn, sheet_name=None, header=None)
            df_tsscn = pd.concat([sheets_tsscn[sheet].rename(columns={0: sheet}) for sheet in sheets_tsscn], axis=1)
            #df = pd.read_excel(fpath, header=None)
            #df_tsscn = pd.read_excel(fpath_tsscn, header=None)
            df.to_hdf(fpath_out, index=False, key="df")
            df_tsscn.to_hdf(fpath_out, index=False, key="df_tsscn")
