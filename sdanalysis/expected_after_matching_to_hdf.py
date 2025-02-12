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
        if "expected_after_match_scn.xlsx" in file:
            continue  # handle this file once the other one is handled
        if "expected_after_match.xlsx" in file:
            print(file)
            fpath = os.path.join(root, file)
            fpath_tsscn = os.path.join(root, os.path.splitext(file)[0] + "_scn.xlsx")
            assert os.path.exists(fpath_tsscn)
            fpath_out = os.path.splitext(fpath)[0] + ".hdf5"
            if os.path.exists(fpath_out):
                print(f"{fpath_out} already exists, skipping...")
                continue
            df = pd.read_excel(fpath, header=None)
            df_tsscn = pd.read_excel(fpath_tsscn, header=None)
            df.to_hdf(fpath_out, index=False, key="df")
            df_tsscn.to_hdf(fpath_out, index=False, key="df_tsscn")
