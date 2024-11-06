import argparse
import numpy as np
import h5py
import custom_io as cio
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib as mpl
from math import floor, ceil, sqrt, atan2, acos, pi, sin, cos
from datetime import datetime
import json
import scipy
from scipy import ndimage
from scipy.spatial import distance_matrix
# for statistical testing on directionality
from scipy.stats import circmean, circstd
import data_documentation
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
import seaborn as sns
# multiprocessing does not work with IPython. Use fork instead.
import multiprocess as mp
import os
import random  # for surrogate algorithm
from collections.abc import Iterable
import math
from functools import partial
from typing import Optional, List
from env_reader import read_env
import warnings


def set_plotting_params():
    mpl.rcParams.update({'font.size': 20})
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    color_palette = sns.color_palette("deep")


def get_directionality_files_list(folder: str) -> List[str]:
    files_list = []
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if "_directionality.h5" in fname:
                files_list.append(os.path.join(root, fname))
    return files_list


def get_uuid_for_directionality_files(files_list: List[str], dd: data_documentation.DataDocumentation) -> dict:
    # given the analysis files (from get_directionality_files_list above),
    # use file name to search for recording uuid.
    uuid_dict = dict()
    for fname in files_list:
        # original nd2 filename should be in the file name as <nd_fname>_<analysis_datetime YYYYMMDD-HHMMSS>_grid.h5
        nd2_fname = "_".join(fname.split("_")[:-2]) + ".nd2"
        uuid = dd.getUUIDForFile(nd2_fname)
        uuid_dict[fname] = uuid
    assert len(uuid_dict.keys()) == len(files_list)
    return uuid_dict


def get_exp_type_for_directionality_files(files_list: List[str], dd: data_documentation.DataDocumentation) -> dict:
    # given the analysis files (from get_directionality_files_list above),
    # use file name to search for recording uuid.
    exp_type_dict = dict()
    for fname in files_list:
        # original nd2 filename should be in the file name as <nd_fname>_<analysis_datetime YYYYMMDD-HHMMSS>_grid.h5
        nd2_fname = "_".join(fname.split("_")[:-2]) + ".nd2"
        exp_type = dd.getExperimentTypeForFile(nd2_fname)
        exp_type_dict[fname] = exp_type
    assert len(exp_type_dict.keys()) == len(files_list)
    return exp_type_dict


def directionality_files_to_df(files_list: List[str], dd: data_documentation.DataDocumentation) -> pd.DataFrame:
    uuid_dict = get_uuid_for_directionality_files(files_list, dd)
    exp_type_dict = get_exp_type_for_directionality_files(files_list, dd)
    df_id_uuid = dd.getIdUuid()
    # to get proper shape of DataFrame, read the first file and keep concatenating to it
    all_onsets_df = pd.read_hdf(files_list[0])
    all_onsets_df["uuid"] = uuid_dict[files_list[0]]
    all_onsets_df["mouse_id"] = df_id_uuid[df_id_uuid["uuid"]
                                           == uuid_dict[files_list[0]]]["mouse_id"].values[0]
    all_onsets_df["exp_type"] = exp_type_dict[files_list[0]]
    assert all_onsets_df["uuid"].isna().sum() == 0
    for fpath in files_list[1:]:
        df = pd.read_hdf(fpath)
        df["uuid"] = uuid_dict[fpath]
        df["mouse_id"] = df_id_uuid[df_id_uuid["uuid"]
                                    == uuid_dict[fpath]]["mouse_id"].values[0]
        df["exp_type"] = exp_type_dict[fpath]
        assert df["uuid"].isna().sum() == 0
        assert df["exp_type"].isna().sum() == 0
        all_onsets_df = pd.concat([all_onsets_df, df])
    return all_onsets_df


def replace_outliers(df: pd.DataFrame, colname: str = "onset_sz", percent: float = 0.05, replace_value=np.NaN) -> pd.DataFrame:
    """Remove the percent*100% of the onset data with the onset values most deviant from group median (excluding NaNs), replacing them with replace_value. 
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 
    colname : str, optional
        the column on which to perform outlier removal, by default "onset_sz"
    percent : float, optional
        The percent of population to replace, by default 0.05
    replace_value : optional

    Returns
    -------
    pd.DataFrame
        The dataframe with outliers replaced with np.nan. Shape of the dataframe is preserved.
    """
    if colname not in df.keys():
        raise ValueError(f"Column {colname} not found in dataframe")
    median_colname = df[colname].dropna().median()
    deviations = np.abs(df[colname] - median_colname)
    deviations_nonan = np.abs(df[colname].dropna() - median_colname)
    if len(deviations_nonan) == 0:  # only NaNs, skip outlier removal
        return df
    # sort in descending order
    deviations_nonan_sorted_desc = np.flip(np.sort(deviations_nonan))
    # get deviation value corresponding to threshold
    deviation_threshold = deviations_nonan_sorted_desc[ceil(
        percent*len(deviations_nonan_sorted_desc))]
    if np.sum(deviations > deviation_threshold) == 0:
        warnings.warn(
            f"Outlier removal failed due to no values exceeding threshold {deviation_threshold}")
        return df
    df.loc[deviations > deviation_threshold, colname] = np.nan
    return df


def replace_multiple_outliers(df: pd.DataFrame, colnames: str = "onset_sz", percent: float = 0.05, replace_value=np.NaN) -> pd.DataFrame:
    for colname in colnames:
        replace_col_outliers = partial(
            replace_outliers, colname=colname, percent=percent, replace_value=replace_value)
        df = df.groupby("uuid_extended").apply(replace_col_outliers)
    return df


def create_seizure_uuid(row):
    """Given a pandas dataframe row, use uuid and i_sz columns to generate a unique identifier for the seizure.
    Parameters
    ----------
    row : pd.Series
        The row of a dataframe
    Returns
    -------
    str
        The unique identifier for the seizure (in form, for sz #0: <uuid>_1, for sz #1: <uuid>_2, etc.). if i_sz < 0, return uuid.
    """
    assert "i_sz" in row.index
    assert "uuid" in row.index
    if pd.isna(row["i_sz"]):
        return row["uuid"]
    elif row["i_sz"] >= 0:
        return row["uuid"] + "_" + str(int(row["i_sz"])+1)


# TODO implement function that creates quantiles_df
def get_quantiles(df_onsets: pd.DataFrame) -> pd.DataFrame:
    # create df with average coordinates per quantile per session (per mouse)
    quantile_dfs = []
    for event_type, quantile_name in zip(["sd1", "sd2", "sz"], ["quantile1", "quantile2", "quantile_sz"]):
        df_q = df_onsets.groupby(
            ["mouse_id", "uuid_extended", quantile_name], as_index=False).mean()
        df_q["quantile_type"] = event_type
        df_q.rename({quantile_name: "quantile"}, axis="columns", inplace=True)
        quantile_dfs.append(df_q)
    return pd.concat(quantile_dfs)


def get_dx_first_last_quantile(df, colname="x"):
    """Assuming df contains unique quantile rows, get the difference between last and first quantile in the colname column of the dataframe df.

    Parameters
    ----------
    df : pd.DataFrame
        A df with quantile and colname columns. The quantile column should contain only unique entries.
    colname : str, optional
        The column for which to get the difference between value for last and first quantile, by default "x"

    Returns
    -------
    df.colname.dtype
        The difference between the value of colname for the last and first quantile.
    """
    max_quantile = df["quantile"].max()
    min_quantile = df["quantile"].min()
    x1 = df[df["quantile"] == max_quantile][colname].values[0]
    x0 = df[df["quantile"] == min_quantile][colname].values[0]
    return x1-x0


def add_polar_coordinates(df: pd.DataFrame, dd: data_documentation.DataDocumentation) -> pd.DataFrame:
    """Given a dataframe with dx and dy columns (i.e. 2D vectors), add polar coordinates (r, theta, theta with aligned windows) to the dataframe. 

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    dd : data_documentation.DataDocumentation
        _description_

    Returns
    -------
    pd.DataFrame
        The original dataframe with three new columns: r, theta (the polar coordinates directly calculated from dx and dy),
        and theta_inj_top, the theta angle corrected for the injection direction (when it is aligned to "top", i.e. positive y, for all entries)
    """
    # for contralateral injections, injection is always on the medial side of the window:
    dict_signs = {"bottom": -1, "top": 1, "same": 1}
    df["r"] = df.apply(lambda row: sqrt(
        pow(row["dx"], 2) + pow(row["dy"], 2)), axis=1)
    df["theta"] = df.apply(lambda row: atan2(row["dy"], row["dx"]), axis=1)
    # correct angle s.t. top direction is always towards injection
    df["theta_inj_top"] = df.apply(lambda row: dict_signs[dd.getInjectionDirection(
        row["mouse_id"])]*row["theta"], axis=1)
    return df


def merge_recording_uuids(df_quantiles: pd.DataFrame) -> pd.DataFrame:
    """In some scenarios, seizure and consequent SD waves might have happened in two separate recordings.
    Change the uuid_extended column of the first recording to match the second recording 
    Parameters
    ----------
    df_quantiles : pd.DataFrame
        The dataframe with quantiles, containing uuid_extended column
    Returns
    -------
    pd.DataFrame
        the dataframe with uuid_extended column modified
    """
    # following two recordings contain 1 seizure-sd event
    df_quantiles["uuid_extended"] = df_quantiles["uuid_extended"].replace(
        "65bff16a4cf04930a5cb14f489a8f99b", "30dc55d1a5dc4b0286d132e72f208ca6")
    return df_quantiles


def get_angles(df_quantiles: pd.DataFrame, drop_na: bool = False) -> pd.DataFrame:
    dict_angles = {"mouse_id": [], "uuid_extended": [],
                   "angle_type": [], "angle": [], "cos_angle": []}
    session_groups = df_quantiles.groupby("uuid_extended")
    for uuid, group in session_groups:
        sz_row = group[group["quantile_type"] == "sz"]
        sd1_row = group[group["quantile_type"] == "sd1"]
        sd2_row = group[group["quantile_type"] == "sd2"]
        sz_len = sz_row["r"].values[0] if len(
            sz_row["r"].values) > 0 else np.nan
        sd1_len = sd1_row["r"].values[0] if len(
            sd1_row["r"].values) > 0 else np.nan
        sd2_len = sd2_row["r"].values[0] if len(
            sd2_row["r"].values) > 0 else np.nan
        if len(sz_row) == 0:  # no seizure for the session
            # sz-sd1
            dict_angles["mouse_id"].append(sd1_row["mouse_id"].values[0])
            dict_angles["uuid_extended"].append(uuid)
            dict_angles["angle_type"].append("sz-sd1")
            dict_angles["angle"].append(np.nan)
            dict_angles["cos_angle"].append(np.nan)
            # sz-sd2
            dict_angles["mouse_id"].append(sd1_row["mouse_id"].values[0])
            dict_angles["uuid_extended"].append(uuid)
            dict_angles["angle_type"].append("sz-sd2")
            dict_angles["angle"].append(np.nan)
            dict_angles["cos_angle"].append(np.nan)
            # sd1-sd2
            if len(sd1_row["dx"].values) > 0 and len(sd2_row["dx"].values) > 0:
                sd1_dot_sd2 = sd1_row["dx"].values[0] * sd2_row["dx"].values[0] + \
                    sd1_row["dy"].values[0] * sd2_row["dy"].values[0]
                costheta12 = sd1_dot_sd2/(sd1_len*sd2_len)
            else:
                sd1_dot_sd2 = np.nan
                costheta12 = np.nan
            dict_angles["mouse_id"].append(sd1_row["mouse_id"].values[0])
            dict_angles["uuid_extended"].append(uuid)
            dict_angles["angle_type"].append("sd1-sd2")
            dict_angles["angle"].append(acos(costheta12))
            dict_angles["cos_angle"].append(costheta12)
        else:
            # u.v = |u|*|v|*cos(theta)
            if len(sz_row["dx"].values) > 0 and len(sd1_row["dx"].values) > 0:
                sz_dot_sd1 = sz_row["dx"].values[0] * sd1_row["dx"].values[0] + \
                    sz_row["dy"].values[0] * sd1_row["dy"].values[0]
                costheta1 = sz_dot_sd1/(sz_len*sd1_len)
            else:
                sz_dot_sd1 = np.nan
                costheta1 = np.nan
            if len(sd2_row["dx"].values) > 0 and len(sd1_row["dx"].values) > 0:
                sz_dot_sd2 = sz_row["dx"].values[0] * sd2_row["dx"].values[0] + \
                    sz_row["dy"].values[0] * sd2_row["dy"].values[0]
                costheta2 = sz_dot_sd2/(sz_len*sd2_len)
            else:
                sz_dot_sd2 = np.nan
                costheta2 = np.nan

            # add sd1-sd2 angle too
            if len(sd1_row["dx"].values) > 0 and len(sd2_row["dx"].values) > 0:
                sd1_dot_sd2 = sd1_row["dx"].values[0] * sd2_row["dx"].values[0] + \
                    sd1_row["dy"].values[0] * sd2_row["dy"].values[0]
                costheta12 = sd1_dot_sd2/(sd1_len*sd2_len)
            else:
                sd1_dot_sd2 = np.nan
                costheta12 = np.nan
            # sz-sd1
            dict_angles["mouse_id"].append(sz_row["mouse_id"].values[0])
            dict_angles["uuid_extended"].append(uuid)
            dict_angles["angle_type"].append("sz-sd1")
            dict_angles["angle"].append(acos(costheta1))
            dict_angles["cos_angle"].append(costheta1)
            # sz-sd2
            dict_angles["mouse_id"].append(sz_row["mouse_id"].values[0])
            dict_angles["uuid_extended"].append(uuid)
            dict_angles["angle_type"].append("sz-sd2")
            dict_angles["angle"].append(acos(costheta2))
            dict_angles["cos_angle"].append(costheta2)
            # sd1-sd2
            dict_angles["mouse_id"].append(sz_row["mouse_id"].values[0])
            dict_angles["uuid_extended"].append(uuid)
            dict_angles["angle_type"].append("sd1-sd2")
            dict_angles["angle"].append(acos(costheta12))
            dict_angles["cos_angle"].append(costheta12)
    df_angles = pd.DataFrame(dict_angles)
    df_angles["angle_deg"] = df_angles["angle"].apply(lambda x: x*180./pi)
    df_angles = df_angles.sort_values(by=["mouse_id", "uuid_extended"])
    if drop_na:
        df_angles = df_angles.dropna()
    # add new column "event index" - the index of the event (Sz, SD1, SD2) in the session
    df_angles["event_index"] = df_angles.groupby(
        'uuid_extended', sort=False).ngroup() + 1
    return df_angles.reset_index(drop=True)


def get_dataset_type(uuids_list: List[str], dd: data_documentation.DataDocumentation) -> str:
    """Given the uuids the dataset consists of, determine if it is stim, tmev, mixed, or other (unknown) dataset. 
    For class definitions, see Returns section.

    Parameters
    ----------
    uuids_list : List[str]
        _description_
    dd : data_documentation.DataDocumentation
        _description_

    Returns
    -------
    str
        One of the following strings: "stim", "tmev", "mixed", "unknown". 
        If all recordings contain only the keywords "chr2" or "jrgeco", or only "tmev", the output category is "stim" or "tmev", respectively.
        If the dataset contains both "chr2" or "jrgeco and "tmev" recordings, the output category is "mixed".
        If the dataset contains at least one unknown UUID, the output category is "unknown".
    """
    contains_stim = False  # chr2 or jrgeco
    contains_tmev = False
    for uuid in uuids_list:
        try:
            exp_type = dd.getExperimentTypeForUuid(uuid)
            if "chr2" in exp_type or "jrgeco" in exp_type:
                contains_stim = True
            if "tmev" in exp_type:
                contains_tmev = True
            if "bilat" in exp_type:
                warnings.warn(
                    f"Bilateral stim recording found! This is currently not supported in analysis \
                        (as no bilateral stim recording has corresponding imaging file). Returning 'unknown' category.")
                return "unknown"
        except KeyError:
            warnings.warn(f"Dataset contains unknown UUID: {uuid}")
            return "unknown"
    if contains_stim and contains_tmev:
        return "mixed"
    elif contains_stim:
        return "stim"
    elif contains_tmev:
        return "tmev"


def main(folder: Optional[str], save_data: bool = False, save_figs: bool = False, file_format: str = "pdf"):
    # TODO: option to choose output file format: excel (xlsx) vs hdf5
    # get datetime for output file name
    output_dtime = cio.get_datetime_for_fname()
    replace_outliers = True  # TODO: add it as a command line argument
    env_dict = read_env()
    set_plotting_params()
    dd = data_documentation.DataDocumentation.from_env_dict(env_dict)
    if save_data:
        output_folder = env_dict["OUTPUT_FOLDER"]
    else:
        output_folder = None
    if folder is None or not os.path.exists(folder):
        folder = cio.open_dir("Open directory with directionality data")
    analysis_fpaths = get_directionality_files_list(folder)
    df_onsets = directionality_files_to_df(analysis_fpaths, dd)
    # get dataset type
    dataset_type = get_dataset_type(df_onsets["uuid"].unique(), dd)
    # for old files, "i_sz" is not a column, as only one seizure per recording was found. Add this column here
    if "i_sz" not in df_onsets.columns:
        df_onsets["i_sz"] = np.nan
    # make a uuid unique to seizure, call the column "uuid_extended"
    df_onsets["uuid_extended"] = df_onsets.apply(create_seizure_uuid, axis=1)
    if replace_outliers:
        df_onsets = replace_multiple_outliers(
            df_onsets, ["onset1", "onset2", "onset_sz"], percent=0.05)
    df_quantiles = get_quantiles(df_onsets)
    # Move from nd2/video-style coordinate system (top left = (0, 0)) to usual plotting coordinate style (bottom left = (0, 0))
    df_quantiles["y_mirrored"] = df_quantiles.apply(
        lambda row: -1*row["y"], axis=1)
    # add dx and dy columns
    df_quantiles = df_quantiles.groupby(["uuid_extended", "quantile_type"], as_index=False).apply(lambda group_df: group_df.assign(
        dx=lambda gdf: get_dx_first_last_quantile(gdf, "x"), dy=lambda gdf: get_dx_first_last_quantile(gdf, "y_mirrored")))
    # throw away duplicate columns
    df_quantiles = df_quantiles[df_quantiles["quantile"] == 0].drop(
        labels=["quantile", "x", "y"], axis=1)
    # add polar coordinates
    df_quantiles = add_polar_coordinates(df_quantiles, dd)
    # merge recording uuids for recordings with seizure-sd events split into two recordings
    df_quantiles = merge_recording_uuids(df_quantiles)
    df_angles = get_angles(df_quantiles, drop_na=True)
    # Create aggregate dataset with single data point = within-mouse average
    df_angles_aggregate = df_angles.replace({"sz-sd1": "Sz-SD1", "sz-sd2": "Sz-SD2", "sd1-sd2": "SD1-SD2"}).rename(
        columns={"mouse_id": "mouse ID"}).sort_values(by="mouse ID").groupby(["mouse ID", "angle_type"]).apply(lambda g: g["angle_deg"].mean())
    # in addition to grouped-by columns (reset_index() makes them columns from indices), keep only mean_angle_deg:
    df_angles_aggregate = pd.DataFrame(df_angles_aggregate, columns=[
                                       "mean_angle_deg"]).reset_index()

    if save_data:
        output_fpath = os.path.join(
            output_folder, f"directionality_{dataset_type}_{output_dtime}.xlsx")
        # output_fpath_agg = os.path.join(
        #    output_folder, f"directionality_{dataset_type}_aggregate_{output_dtime}.xlsx")
        df_angles.to_excel(output_fpath, index=False)
        df_angles_aggregate.to_excel(os.path.join(
            output_folder, f"directionality_aggregate_{dataset_type}_{output_dtime}.xlsx"), index=False)
    # TODO: add surrogate sampling option, export data
    # TODO: modify all_onsets_df by adding seizure onset speed

    return (df_angles, df_angles_aggregate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None,
                        help="Folder with directionality data h5 files", )
    parser.add_argument("--save_data", action="store_true",
                        help="Save data to Excel file")
    parser.add_argument("--save_figs", action="store_true",
                        help="Save figures")
    parser.add_argument("--file_format", type=str, default="pdf",
                        help="File format for figures")
    args = parser.parse_args()
    main(args.folder, args.save_data, args.save_figs, args.file_format)
