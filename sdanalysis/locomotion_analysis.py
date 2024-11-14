"""
locomotion_analysis.py - Locomotion analysis pipeline for TMEV, window and cannula stimulation datasets.
"""
from typing import Tuple, Optional
import argparse
from collections import OrderedDict
import sdanalysis.custom_io as cio
import sdanalysis.env_reader
import h5py
import numpy as np
import os
import pandas as pd
import seaborn as sns
import seaborn as sns
from locomotion_functions import (
    apply_threshold,
    get_episodes,
    calculate_avg_speed,
    calculate_max_speed,
    get_trace_delta
)
import data_documentation

# Define metrics
stat_metrics = [
    "totdist_abs_norm",
    "running%",
    "running_episodes",
    "avg_speed",
    "running_episodes_mean_length",
    "max_speed",
]  # metrics to test for
# Define metric labels
dict_metric_label = OrderedDict(
    [
        ("totdist_abs", "Total (absolute) distance, a.u."),
        ("running%", "% of time spent with locomotion"),
        ("running_episodes", "Number of running episodes"),
        ("avg_speed", "Average of locomotion velocity"),
        ("running_episodes_mean_length", "Mean length of running episodes, a.u."),
        ("max_speed", "Max velocity of locomotion, a.u."),
    ]
)

# TODO: add figure saving


def main(
    fpath: Optional[str],
    ampl_threshold: float = 0.2,
    temp_threshold: int = 15,
    episode_merge_threshold: int = 8,
    save_data: bool = True,
    save_figs: bool = False,
    file_format: str = "pdf",
    save_sanity_check: bool = False,
    save_waterfall: bool = False,
) -> Tuple[pd.DataFrame]:
    """Run the locomotion analysis pipeline. Optionally save output data and figures.

    Parameters
    ----------
    ampl_threshold : float, optional
        threshold that one element within the running episode candidate has to be reached for the episode to not be discarded, by default 0.2
    temp_threshold : int, optional
        in number of frames. In 15 Hz, this amounts to 1 s threshold that a candidate episode has to reach to not be discarded, by default 15
    episode_merge_threshold : int, optional
        merge running episodes if temporal distance distance smaller than this many frames or equal (15 Hz!), by default 8
    save_data : bool, optional
        _description_, by default True
    save_figs : bool, optional
        _description_, by default False
    file_format : str, optional
        _description_, by default "pdf"
    save_sanity_check : bool, optional
        _description_, by default False
    save_waterfall : bool, optional
        _description_, by default False

    Returns
    -------
    Tuple[pd.DataFrame]
        The dataframes containing the results of the locomotion analysis:
        [0]: dataframe with individual metrics for each segment in each recording,
        [1]: with the differences for each recording,
        [2]: with the differences aggregated for each mouse.
    """
    # TODO: option to choose output file format: excel (xlsx) vs hdf5?
    if fpath is None: 
        raise ValueError("No dataset file path provided!")
    elif not os.path.exists(fpath):
        raise FileNotFoundError(f"Dataset file not found at\n\t{fpath}")
    else:
        assembled_traces_fpath = fpath
    # TODO: add as parameter, or remove completely (matlab)?
    save_as_workspace = False
    if save_figs:
        print(f"Going to save figures as {file_format} files.")
    sns.set(font_scale=3)
    sns.set_style("whitegrid")

    # Load .env file
    env_dict = sdanalysis.env_reader.read_env()
    data_doc = data_documentation.DataDocumentation.from_env_dict(env_dict)

    # Set up output folder
    output_folder = env_dict["OUTPUT_FOLDER"]
    print(f"Output files will be saved to {output_folder}")
    output_dtime = cio.get_datetime_for_fname()

    # Set up color coding
    df_colors = data_doc.get_colorings()
    dict_colors_mouse = df_colors[["mouse_id", "color"]].to_dict(orient="list")
    dict_colors_mouse = dict(
        zip(dict_colors_mouse["mouse_id"], dict_colors_mouse["color"])
    )
    # Load list of events
    df_events_list = data_doc.get_events_list()
    # Open data

    print(f"Data chosen: {assembled_traces_fpath}")
    # Determine dataset type
    is_win_stim = False
    is_cannula_stim = False
    if "window-stim" in assembled_traces_fpath.lower():
        is_win_stim = True
        print("Window stimulation dataset detected")
    elif "cannula-stim" in assembled_traces_fpath.lower():
        is_cannula_stim = True
        print("Cannula stim dataset detected")
    else:
        print("TMEV dataset detected")
    # define mice to use
    # TODO: add NC ChR2 separately
    if is_win_stim:
        used_mouse_ids = ["OPI-2239", "WEZ-8917", "WEZ-8924", "WEZ-8922"]
    elif is_cannula_stim:
        used_mouse_ids = ["WEZ-8946", "WEZ-8960", "WEZ-8961"]
    dataset_type = (
        "win-stim" if is_win_stim else "cannula-stim" if is_cannula_stim else "tmev"
    )

    # Load the traces
    traces_dict = dict()
    traces_meta_dict = dict()
    # first keys (groups in the file) are event uuids, inside the following dataset names:
    # 'lfp_mov_t', 'lfp_mov_y', 'lfp_t', 'lfp_y', 'lv_dist', 'lv_rounds',
    # 'lv_running', 'lv_speed', 'lv_t_s', 'lv_totdist', 'mean_fluo'
    with h5py.File(assembled_traces_fpath, "r") as hf:
        for uuid in hf.keys():
            if (not is_win_stim) or (hf[uuid].attrs["mouse_id"] in used_mouse_ids):
                session_dataset_dict = dict()
                session_meta_dict = dict()
                for dataset_name in hf[uuid].keys():
                    session_dataset_dict[dataset_name] = np.array(
                        hf[uuid][dataset_name]
                    )
                for attr_name in hf[uuid].attrs:
                    session_meta_dict[attr_name] = hf[uuid].attrs[attr_name]
                traces_dict[uuid] = session_dataset_dict.copy()
                traces_meta_dict[uuid] = session_meta_dict.copy()
    # merge chr2_ctl_unilat and chr2_ctl_bilat into single category chr2_ctl
    merge_ctl = True
    if merge_ctl:
        for uuid, meta in traces_meta_dict.items():
            if "chr2_ctl" in meta["exp_type"]:
                traces_meta_dict[uuid]["exp_type"] = "chr2_ctl"

    # Get overall speed range for plotting/scaling
    min_speed = np.inf
    max_speed = -np.inf
    for event_uuid in traces_dict:
        speed = traces_dict[event_uuid]["lv_speed"]
        min_candidate = np.min(speed)
        max_candidate = np.max(speed)
        if min_candidate < min_speed:
            min_speed = min_candidate
        if max_candidate > max_speed:
            max_speed = max_candidate
    print(f"Speed range: {min_speed} to {max_speed}")
    lv_speed_amp = max_speed - min_speed
    # Similarly, get overall fluorescence range (if available)
    if not is_cannula_stim:
        min_fluo = np.inf
        max_fluo = -np.inf
        for event_uuid in traces_dict.keys():
            mean_fluo = traces_dict[event_uuid]["mean_fluo"]
            if is_win_stim:
                if traces_meta_dict[event_uuid]["mouse_id"] in used_mouse_ids:
                    if "i_stim_begin_frame" in traces_meta_dict[event_uuid].keys():
                        # get 0-indexing, inclusive first and last frames of stim
                        i_begin_stim = traces_meta_dict[event_uuid][
                            "i_stim_begin_frame"
                        ]
                        i_end_stim = traces_meta_dict[event_uuid]["i_stim_end_frame"]
                        mean_fluo_except_stim = np.concatenate(
                            [mean_fluo[:i_begin_stim], mean_fluo[i_end_stim + 1:]]
                        )
                        min_candidate = np.min(mean_fluo_except_stim)
                        max_candidate = np.max(mean_fluo_except_stim)
                    else:
                        print(f"{event_uuid} missing i_stim_begin_frame!")
            else:
                min_candidate = np.min(mean_fluo)
                max_candidate = np.max(mean_fluo)
            if min_candidate < min_fluo:
                min_fluo = min_candidate
            if max_candidate > max_fluo:
                max_fluo = max_candidate
        print(f"{min_fluo} to {max_fluo}")

    # Calculate locomotion statistics
    use_manual_bl_am_length = True
    # tmev and chr2: 4500, bilat stim: 4425 [this used to be 4486, now shortest recording has length equivalent to only 4425 frames]
    bl_manual_length = 4425 if is_cannula_stim else 4500
    am_manual_length = bl_manual_length

    # each entry (row) should have columns:
    # uuid of event, mouse id, window type, segment type (bl/sz/am), segment length in frames, totdist, running, speed
    list_statistics = []
    dict_episodes = {}
    # contains the post-filtering "running" trace, of which the running% is calculated (divided by segment length)
    loco_binary_traces = {}
    loco_episodes = {}  # contains the first and last indices of the locomotion episodes
    begin_end_frames_dict = {}

    for event_uuid in traces_dict:
        mouse_id = traces_meta_dict[event_uuid]["mouse_id"]
        win_type = traces_meta_dict[event_uuid]["window_type"]
        # get segment lengths
        n_bl_frames = traces_meta_dict[event_uuid]["n_bl_frames"]
        n_am_frames = traces_meta_dict[event_uuid]["n_am_frames"]
        n_frames = traces_meta_dict[event_uuid]["n_frames"]
        n_sz_frames = n_frames - n_am_frames - n_bl_frames

        if use_manual_bl_am_length:
            if (bl_manual_length > n_bl_frames) or (am_manual_length > n_am_frames):
                print(
                    f"{mouse_id} {event_uuid}:\n\tNot enough bl ({n_bl_frames}, {bl_manual_length} required) or am ({n_am_frames}, {am_manual_length} required) frames available. Skipping..."
                )
                continue
            # todo: set first and last frames for bl and am (as well as sz). If not use_manual_bl_am_length, also set it!
            # then modify code below to first and last frames
            else:
                # define baseline as last frame before sz segment, and starting bl_manual_length frames before
                last_frame_bl = n_bl_frames - 1  # 0 indexing: last bl frame, inclusive
                first_frame_bl = last_frame_bl - bl_manual_length + 1  # inclusive
                assert first_frame_bl >= 0
                # define aftermath as first frame after sz segment, and ending am_manual_length frames after
                first_frame_am = n_bl_frames + n_sz_frames  # inclusive
                assert first_frame_am == n_frames - n_am_frames

                last_frame_am = first_frame_am + am_manual_length - 1  # inclusive

                # convert to [begin, end), i.e. left inclusive, right exclusive, for numpy indexing
                last_frame_bl += 1
                last_frame_am += 1

        else:
            first_frame_bl = 0  # inclusive
            last_frame_bl = n_bl_frames  # exclusive

            first_frame_am = n_bl_frames + n_sz_frames  # inclusive
            last_frame_am = n_frames  # exclusive

        begin_end_frames_dict[event_uuid] = [
            first_frame_bl,
            last_frame_bl,
            first_frame_am,
            last_frame_am,
        ]

        # print(f"{ddoc.getNikonFileNameForUuid(event_uuid)}:\n\t{n_bl_frames} bl, {n_sz_frames} mid, {n_am_frames} am")
        # get movement data
        lv_totdist = traces_dict[event_uuid]["lv_totdist"]
        lv_totdist_abs = traces_dict[event_uuid]["lv_totdist_abs"]
        lv_running = traces_dict[event_uuid]["lv_running"]
        lv_speed = traces_dict[event_uuid]["lv_speed"]

        # apply post-processing threshold to "running"

        # cut up data into segments
        lv_totdist_bl = lv_totdist[first_frame_bl:last_frame_bl]
        lv_totdist_sz = lv_totdist[last_frame_bl:first_frame_am]
        lv_totdist_am = lv_totdist[first_frame_am:last_frame_am]
        if not use_manual_bl_am_length:
            assert len(lv_totdist_bl) + len(lv_totdist_sz) + len(lv_totdist_am) == len(
                lv_totdist
            )
        else:
            assert len(lv_totdist_bl) == bl_manual_length
            assert len(lv_totdist_am) == am_manual_length

        lv_totdist_abs_bl = lv_totdist_abs[first_frame_bl:last_frame_bl]
        lv_totdist_abs_sz = lv_totdist_abs[last_frame_bl:first_frame_am]
        lv_totdist_abs_am = lv_totdist_abs[first_frame_am:last_frame_am]

        lv_running_bl = lv_running[first_frame_bl:last_frame_bl]
        lv_running_sz = lv_running[last_frame_bl:first_frame_am]
        lv_running_am = lv_running[first_frame_am:last_frame_am]

        lv_speed_bl = lv_speed[first_frame_bl:last_frame_bl]
        lv_speed_sz = lv_speed[last_frame_bl:first_frame_am]
        lv_speed_am = lv_speed[first_frame_am:last_frame_am]

        # if not (np.all(lv_totdist[1:] >= lv_totdist[:-1]) or np.all(lv_totdist[1:] <= lv_totdist[:-1])):
        #    print(f"Not monotonous: {mouse_id} {event_uuid}")
        # calculate statistics
        # last_frame is exclusive, i.e. [begin, end)
        totdist_bl = get_trace_delta(lv_totdist, first_frame_bl, last_frame_bl)
        totdist_sz = get_trace_delta(lv_totdist, last_frame_bl, first_frame_am)
        totdist_am = get_trace_delta(lv_totdist, first_frame_am, last_frame_am)

        totdist_abs_bl = get_trace_delta(
            lv_totdist_abs, first_frame_bl, last_frame_bl)
        totdist_abs_sz = get_trace_delta(
            lv_totdist_abs, last_frame_bl, first_frame_am)
        totdist_abs_am = get_trace_delta(
            lv_totdist_abs, first_frame_am, last_frame_am)

        speed_bl = sum(lv_speed_bl)
        speed_sz = sum(lv_speed_sz)
        speed_am = sum(lv_speed_am)
        # calculate average speed
        lv_speed_bl = np.array(lv_speed_bl)
        lv_speed_sz = np.array(lv_speed_sz)
        lv_speed_am = np.array(lv_speed_am)
        lv_running_bl = np.array(lv_running_bl)
        lv_running_sz = np.array(lv_running_sz)
        lv_running_am = np.array(lv_running_am)
        # take absolute values!
        avg_speed_bl = calculate_avg_speed(lv_speed_bl, lv_speed_bl > 0)
        avg_speed_sz = calculate_avg_speed(lv_speed_sz, lv_speed_sz > 0)
        avg_speed_am = calculate_avg_speed(lv_speed_am, lv_speed_am > 0)
        # take absolute max speed!
        max_speed_bl = calculate_max_speed(lv_speed_bl)
        max_speed_sz = calculate_max_speed(lv_speed_sz)
        max_speed_am = calculate_max_speed(lv_speed_am)

        # number of running episodes, length
        # 15 frames in 15 Hz is 1 s.
        list_episodes_bl = get_episodes(
            lv_running_bl, True, episode_merge_threshold, return_begin_end_frames=True
        )
        list_episodes_sz = get_episodes(
            lv_running_sz, True, episode_merge_threshold, return_begin_end_frames=True
        )
        list_episodes_am = get_episodes(
            lv_running_am, True, episode_merge_threshold, return_begin_end_frames=True
        )

        # apply a filter to episodes, discard those that do not fulfill the criteria
        list_episodes_bl = apply_threshold(
            lv_speed_bl,
            list_episodes_bl,
            temp_threshold,
            ampl_threshold,
        )
        list_episodes_sz = apply_threshold(
            lv_speed_sz,
            list_episodes_sz,
            temp_threshold,
            ampl_threshold,
        )
        list_episodes_am = apply_threshold(
            lv_speed_am,
            list_episodes_am,
            temp_threshold,
            ampl_threshold,
        )

        # get the episode lengths and number of episodes
        list_episode_lengths_bl = [
            ep[1] - ep[0] + 1 for ep in list_episodes_bl]
        n_episodes_bl = len(list_episodes_bl)

        list_episode_lengths_sz = [
            ep[1] - ep[0] + 1 for ep in list_episodes_sz]
        n_episodes_sz = len(list_episode_lengths_sz)

        list_episode_lengths_am = [
            ep[1] - ep[0] + 1 for ep in list_episodes_am]
        n_episodes_am = len(list_episode_lengths_am)

        # apply filtering to "running" signal

        filtered_running_bl = np.zeros(
            len(lv_running_bl), dtype=lv_running_bl.dtype)
        filtered_running_sz = np.zeros(
            len(lv_running_sz), dtype=lv_running_sz.dtype)
        filtered_running_am = np.zeros(
            len(lv_running_am), dtype=lv_running_am.dtype)
        # add zeros before and after segments to match original recording length
        filtered_running_prebl = np.zeros(
            first_frame_bl, dtype=lv_running_bl.dtype)
        filtered_running_postam = np.zeros(
            len(lv_totdist) - last_frame_am, dtype=lv_running_am.dtype
        )

        for episode in list_episodes_bl:
            filtered_running_bl[episode[0]: episode[1] + 1] = 1
        for episode in list_episodes_sz:
            filtered_running_sz[episode[0]: episode[1] + 1] = 1
        for episode in list_episodes_am:
            filtered_running_am[episode[0]: episode[1] + 1] = 1

        # create "running" statistic, using filtered data
        running_bl = np.sum(filtered_running_bl)  # np.sum(lv_running_bl)
        running_sz = np.sum(filtered_running_sz)  # np.sum(lv_running_sz)
        running_am = np.sum(filtered_running_am)  # np.sum(lv_running_am)

        loco_binary_traces[event_uuid] = np.concatenate(
            [
                filtered_running_prebl,
                filtered_running_bl,
                filtered_running_sz,
                filtered_running_am,
                filtered_running_postam,
            ]
        )
        assert len(loco_binary_traces[event_uuid]) == len(lv_totdist)

        # as running already has a built-in merging (see Matlab beltAddRunningProperties.m), we can count the leading edges in that data
        # n_episodes_bl2 = sum((lv_running_bl[1:] - lv_running_bl[:-1]) > 0)
        # n_episodes_sz2 = sum((lv_running_sz[1:] - lv_running_sz[:-1]) > 0)
        # n_episodes_am2 = sum((lv_running_am[1:] - lv_running_am[:-1]) > 0)

        # print(f"bl: {n_episodes_bl} vs {n_episodes_bl2}, sz: {n_episodes_sz} vs {n_episodes_sz2}, am: {n_episodes_am} vs {n_episodes_am2}")

        # add to episodes dict
        if mouse_id not in dict_episodes.keys():
            dict_episodes[mouse_id] = dict()
        dict_episodes[mouse_id][event_uuid] = dict()

        list_episode_lengths_bl = np.array(list_episode_lengths_bl)
        list_episode_lengths_sz = np.array(list_episode_lengths_sz)
        list_episode_lengths_am = np.array(list_episode_lengths_am)

        dict_episodes[mouse_id][event_uuid]["bl"] = list_episode_lengths_bl
        dict_episodes[mouse_id][event_uuid]["sz"] = list_episode_lengths_sz
        dict_episodes[mouse_id][event_uuid]["am"] = list_episode_lengths_am

        # calculate mean episode length, std
        bl_episode_mean_len = (
            list_episode_lengths_bl.mean() if len(list_episode_lengths_bl) > 0 else 0
        )
        sz_episode_mean_len = (
            list_episode_lengths_sz.mean() if len(list_episode_lengths_sz) > 0 else 0
        )
        am_episode_mean_len = (
            list_episode_lengths_am.mean() if len(list_episode_lengths_am) > 0 else 0
        )

        bl_episode_std = list_episode_lengths_bl.std()
        sz_episode_std = list_episode_lengths_sz.std()
        am_episode_std = list_episode_lengths_am.std()

        if "exp_type" in traces_meta_dict[event_uuid].keys():
            exp_type = traces_meta_dict[event_uuid]["exp_type"]
        else:
            exp_type = "tmev"

        segment_length_bl = last_frame_bl - first_frame_bl
        segment_length_sz = first_frame_am - last_frame_bl
        segment_length_am = last_frame_am - first_frame_am

        # add to data list
        list_statistics.append(
            [
                event_uuid,
                mouse_id,
                win_type,
                exp_type,
                "bl",
                segment_length_bl,
                totdist_bl,
                totdist_abs_bl,
                running_bl,
                speed_bl,
                avg_speed_bl,
                n_episodes_bl,
                bl_episode_mean_len,
                bl_episode_std,
                max_speed_bl,
            ]
        )
        list_statistics.append(
            [
                event_uuid,
                mouse_id,
                win_type,
                exp_type,
                "sz",
                segment_length_sz,
                totdist_sz,
                totdist_abs_sz,
                running_sz,
                speed_sz,
                avg_speed_sz,
                n_episodes_sz,
                sz_episode_mean_len,
                sz_episode_std,
                max_speed_sz,
            ]
        )
        list_statistics.append(
            [
                event_uuid,
                mouse_id,
                win_type,
                exp_type,
                "am",
                segment_length_am,
                totdist_am,
                totdist_abs_am,
                running_am,
                speed_am,
                avg_speed_am,
                n_episodes_am,
                am_episode_mean_len,
                am_episode_std,
                max_speed_am,
            ]
        )
    df_stats = pd.DataFrame(
        data=list_statistics,
        columns=[
            "event_uuid",
            "mouse_id",
            "window_type",
            "exp_type",
            "segment_type",
            "segment_length",
            "totdist",
            "totdist_abs",
            "running",
            "speed",
            "avg_speed",
            "running_episodes",
            "running_episodes_mean_length",
            "running_episodes_length_std",
            "max_speed",
        ],
    )
    # set NaN to 0 (running_episodes_mean_length: if no episodes, then mean segment length is 0)
    df_stats["running_episodes_mean_length"] = df_stats[
        "running_episodes_mean_length"
    ].fillna(value=0)
    # pick a scale factor for better readability: 0.000513 -> 51.3, for example
    if "n_bl_frames" in locals():
        scale_factor = n_bl_frames  # scale up to bl segment length
    else:
        scale_factor = 10000
    df_stats["totdist_norm"] = (
        scale_factor * df_stats["totdist"] / df_stats["segment_length"]
    )
    df_stats["totdist_abs_norm"] = (
        scale_factor * df_stats["totdist_abs"] / df_stats["segment_length"]
    )
    df_stats["running_norm"] = (
        scale_factor * df_stats["running"] / df_stats["segment_length"]
    )
    df_stats["speed_norm"] = (
        scale_factor * df_stats["speed"] / df_stats["segment_length"]
    )
    # % of time spent running
    # get value as true % instead of [0, 1] float
    df_stats["running%"] = 100.0 * \
        df_stats["running"] / df_stats["segment_length"]
    # replace NaN with 0 in average speed
    df_stats["avg_speed"] = df_stats["avg_speed"].fillna(0)
    # Add color column
    df_stats["color"] = df_stats.apply(
        lambda row: dict_colors_mouse[row["mouse_id"]], axis=1
    )
    dict_colors_event = df_stats[[
        "event_uuid", "color"]].to_dict(orient="list")
    dict_colors_event = dict(
        zip(dict_colors_event["event_uuid"], dict_colors_event["color"])
    )
    # Standardize window type
    df_stats["window_type"] = df_stats["window_type"].replace(
        {"Cx": "NC", "ca1": "CA1"}
    )
    # Aggregate per mouse (mean)
    df_stats_per_mouse_mean = (
        df_stats.drop(columns=["event_uuid", "window_type", "color"], axis=0)
        .groupby(["mouse_id", "exp_type", "segment_type"])
        .agg(func="mean")
        .reset_index()
    )
    df_stats_per_mouse_mean["window_type"] = df_stats_per_mouse_mean.apply(
        lambda row: data_doc.get_mouse_win_inj_info(
            row["mouse_id"]).iloc[0].window_type,
        axis=1,
    )
    df_stats_per_mouse_mean["color"] = df_stats_per_mouse_mean.apply(
        lambda row: df_colors[df_colors["mouse_id"]
                              == row["mouse_id"]].iloc[0].color,
        axis=1,
    )
    # Create identifier that is unique to mouse ID + experiment type combination
    df_stats_per_mouse_mean["mouse_id_exp_type"] = (
        df_stats_per_mouse_mean["mouse_id"] + " " +
        df_stats_per_mouse_mean["exp_type"]
    )
    # Get experiment type-related quantities
    n_exp_types = len(df_stats.exp_type.unique())
    exp_types = df_stats.exp_type.unique()

    # 1. TMEV
    if not is_win_stim and not is_cannula_stim:
        value_mapping = {"bl": "baseline",
                         "sz": "seizure", "am": "post-seizure"}
        df_stats["segment_type"] = df_stats["segment_type"].apply(
            lambda x: value_mapping[x]
        )
        df_stats_ca1 = df_stats[df_stats["window_type"] == "CA1"]
        df_stats_nc = df_stats[df_stats["window_type"] == "NC"]
    # Plot all possible metrics
    if not is_win_stim and not is_cannula_stim:
        df_stats_only_bl_am = df_stats[
            df_stats["segment_type"].isin(
                [value_mapping["bl"], value_mapping["am"]])
        ]
    # aggregate
    if not is_win_stim and not is_cannula_stim:
        df_stats_per_mouse_mean["segment_type"] = df_stats_per_mouse_mean[
            "segment_type"
        ].apply(lambda x: value_mapping[x])
    if not is_win_stim and not is_cannula_stim:
        df_stats_per_mouse_mean_only_bl_am = df_stats_per_mouse_mean[
            df_stats_per_mouse_mean["segment_type"].isin(
                [value_mapping["bl"], value_mapping["am"]]
            )
        ]
    if not is_win_stim and not is_cannula_stim:
        df_stats_per_mouse_mean_only_bl_am = (
            df_stats_per_mouse_mean_only_bl_am.sort_values(
                by=["mouse_id", "exp_type", "segment_type"]
            )
        )

    # 2. ChR2 (bl - stim - (sz) - am protocol)
    # Rename bl -> baseline, am -> post-stimuation/sz
    if is_win_stim or is_cannula_stim:
        value_mapping = {
            "bl": "baseline",
            "sz": "stimulation",
            "am": "post-stimulation/Sz",
        }
    if is_win_stim or is_cannula_stim:
        df_stats["segment_type"] = df_stats["segment_type"].apply(
            lambda x: value_mapping[x]
        )
        df_stats_per_mouse_mean["segment_type"] = df_stats_per_mouse_mean[
            "segment_type"
        ].apply(lambda x: value_mapping[x])
    if is_win_stim or is_cannula_stim:
        df_stats_only_bl_am = df_stats[
            df_stats["segment_type"].isin(
                [value_mapping["bl"], value_mapping["am"]])
        ]
    # Make pair plot for CA1
    df_stats_ca1 = df_stats[df_stats["window_type"] == "CA1"]

    # Make pair plot for NC
    df_stats_nc = df_stats[df_stats["window_type"] == "NC"]

    # No window
    df_stats_nowin = df_stats[df_stats["window_type"] == "None"]

    # Aggregate
    # CA1
    df_stats_per_mouse_mean_ca1 = df_stats_per_mouse_mean[
        df_stats_per_mouse_mean["window_type"] == "CA1"
    ]
    # #df_stats_per_mouse_mean_ca1["segment_type"] = df_stats_per_mouse_mean_ca1["segment_type"].apply(lambda x: value_mapping[x])
    df_stats_per_mouse_mean_ca1_only_bl_am = df_stats_per_mouse_mean_ca1[
        df_stats_per_mouse_mean_ca1["segment_type"].isin(
            [value_mapping["bl"], value_mapping["am"]]
        )
    ]
    df_stats_per_mouse_mean_ca1_only_bl_am = (
        df_stats_per_mouse_mean_ca1_only_bl_am.sort_values(
            by=["mouse_id", "exp_type", "segment_type"]
        )
    )

    # NC
    if len(df_stats_nc) > 0:
        df_stats_per_mouse_mean_nc = df_stats_per_mouse_mean[
            df_stats_per_mouse_mean["window_type"] == "NC"
        ]
        # df_stats_per_mouse_mean_nc["segment_type"] = df_stats_per_mouse_mean_nc["segment_type"].apply(lambda x: value_mapping[x])
        df_stats_per_mouse_mean_nc_only_bl_am = df_stats_per_mouse_mean_nc[
            df_stats_per_mouse_mean_nc["segment_type"].isin(
                [value_mapping["bl"], value_mapping["am"]]
            )
        ]
        df_stats_per_mouse_mean_nc_only_bl_am = (
            df_stats_per_mouse_mean_nc_only_bl_am.sort_values(
                by=["mouse_id", "exp_type", "segment_type"]
            )
        )
        n_exp_types = len(df_stats_per_mouse_mean_nc.exp_type.unique())
        n_cols = len(dict_metric_label.keys())

    # No window
    if len(df_stats_nowin) > 0:
        df_stats_per_mouse_mean_nowin = df_stats_per_mouse_mean[
            df_stats_per_mouse_mean["window_type"] == "None"
        ]
        df_stats_per_mouse_mean_nowin_only_bl_am = df_stats_per_mouse_mean_nowin[
            df_stats_per_mouse_mean_nowin["segment_type"].isin(
                [value_mapping["bl"], value_mapping["am"]]
            )
        ]
        df_stats_per_mouse_mean_nowin_only_bl_am = (
            df_stats_per_mouse_mean_nowin_only_bl_am.sort_values(
                by=["mouse_id", "exp_type", "segment_type"]
            )
        )

    # Waterfall plot & sanity check
    # keys: experiment_type, window_type, mouse_id, value: [uuid1, uuid2, ...]
    exptype_wintype_id_dict = {}
    for uuid in traces_meta_dict.keys():
        if is_win_stim or is_cannula_stim:
            exp_type = traces_meta_dict[uuid]["exp_type"]
        else:
            exp_type = "tmev"
        win_type = traces_meta_dict[uuid]["window_type"]
        mouse_id = traces_meta_dict[uuid]["mouse_id"]
        if exp_type not in exptype_wintype_id_dict.keys():
            exptype_wintype_id_dict[exp_type] = dict()
        if win_type not in exptype_wintype_id_dict[exp_type].keys():
            exptype_wintype_id_dict[exp_type][win_type] = dict()
        if mouse_id not in exptype_wintype_id_dict[exp_type][win_type].keys():
            # list of uuids
            exptype_wintype_id_dict[exp_type][win_type][mouse_id] = []
        exptype_wintype_id_dict[exp_type][win_type][mouse_id].append(uuid)

    df_stats_only_bl_am["avg_speed"] = df_stats_only_bl_am["avg_speed"].fillna(
        0)
    assert df_stats_only_bl_am["avg_speed"].isna().sum() == 0

    df_to_save = df_stats[
        (df_stats["segment_type"].isin(
            [value_mapping["bl"], value_mapping["am"]]))
    ].reset_index(drop=True)
    df_to_save_aggregate = (
        df_stats_per_mouse_mean[
            (
                df_stats_per_mouse_mean["segment_type"].isin(
                    [value_mapping["bl"], value_mapping["am"]]
                )
            )
        ]
        .sort_values(by=["mouse_id", "exp_type", "segment_type"])
        .reset_index(drop=True)
    )

    df_diff = get_differences(df_stats, value_mapping).reset_index(drop=True)
    df_diff_aggregate = get_differences(
        df_stats_per_mouse_mean, value_mapping
    ).reset_index(drop=True)

    if save_data:
        output_fpath = os.path.join(
            output_folder, f"loco_{dataset_type}_{output_dtime}.xlsx"
        )
        output_fpath_agg = os.path.join(
            output_folder, f"loco_{dataset_type}_aggregate_{output_dtime}.xlsx"
        )
        df_to_save.to_excel(output_fpath, index=False)
        df_to_save_aggregate.to_excel(output_fpath_agg, index=False)
        print(
            f"Results exported to \n\t{output_fpath}\nand\n\t{output_fpath_agg}")
        save_differences(df_diff, output_folder, dataset_type, output_dtime)
        save_differences(df_diff_aggregate, output_folder,
                         dataset_type, output_dtime)

    if save_as_workspace:
        save_to_workspace(df_to_save, dataset_type,
                          output_folder, output_dtime)

    return (df_to_save, df_to_save_aggregate, df_diff, df_diff_aggregate)


def get_differences(df_stat_data, value_mapping):
    # in each row, plot for each exp_type the given metric. Plot different metric each row.
    # pre/post values might be paired by event_uuid (individual sessions) or mouse_id (aggregate).
    group_by_colname = "event_uuid"
    if group_by_colname not in df_stat_data.columns:
        group_by_colname = "mouse_id"
    df_differences = None
    if len(df_stat_data) > 0:
        for metric in stat_metrics:
            if group_by_colname == "event_uuid":
                df_metric_pivot = df_stat_data.pivot(
                    columns="segment_type", index=group_by_colname, values=metric
                ).reset_index()
            # mouse_id may not be unique (multiple experiment types, like chr2_ctl, chr2_sd, for one mouse)
            else:
                df_metric_pivot = df_stat_data.pivot(
                    columns="segment_type",
                    index=[group_by_colname, "exp_type"],
                    values=metric,
                ).reset_index()
            # 1 window per mouse
            df_metric_pivot["window_type"] = df_metric_pivot.apply(
                lambda row: df_stat_data[
                    df_stat_data[group_by_colname] == row[group_by_colname]
                ].window_type.iloc[0],
                axis=1,
            )
            df_metric_pivot["mouse_id"] = df_metric_pivot.apply(
                lambda row: df_stat_data[
                    df_stat_data[group_by_colname] == row[group_by_colname]
                ].mouse_id.iloc[0],
                axis=1,
            )
            if "exp_type" not in df_metric_pivot.columns:
                df_metric_pivot["exp_type"] = df_metric_pivot.apply(
                    lambda row: df_stat_data[
                        df_stat_data[group_by_colname] == row[group_by_colname]
                    ].exp_type.iloc[0],
                    axis=1,
                )
            metric_diff_name = f"delta_{metric}"
            df_metric_pivot[metric_diff_name] = (
                df_metric_pivot[value_mapping["am"]]
                - df_metric_pivot[value_mapping["bl"]]
            )
            # only keep the change (delta), drop the quantities themselves
            df_metric_pivot = df_metric_pivot.drop(
                [value_mapping["bl"], value_mapping["sz"],
                    value_mapping["am"]], axis=1
            )
            if df_differences is None:
                df_differences = df_metric_pivot
            else:
                df_differences = df_differences.merge(
                    df_metric_pivot,
                    on=[group_by_colname, "window_type", "exp_type", "mouse_id"],
                )
        return df_differences


def save_differences(df_diffs, output_folder, dataset_type, output_dtime):
    group_by_colname = "event_uuid"
    if group_by_colname not in df_diffs.columns:
        group_by_colname = "mouse_id"
    if group_by_colname == "event_uuid":
        data_output_fpath = os.path.join(
            output_folder, f"loco_{dataset_type}_delta_{output_dtime}.xlsx"
        )
    else:  # aggregate data used
        data_output_fpath = os.path.join(
            output_folder, f"loco_{dataset_type}_aggregate_delta_{output_dtime}.xlsx"
        )
    df_diffs.to_excel(data_output_fpath, index=False)
    print(f"Saved data to {data_output_fpath}")


# Optional: save to matlab workspace
def save_to_workspace(df_to_save, dset_type, output_folder, output_dtime):
    """Save processed data to matlab workspace

    Parameters
    ----------
    df_to_save : pandas.DataFrame
        The dataframe to save
    dset_type : str
        "chr2", "bilat", or "tmev"
    output_folder : str
        The folder to save the data to
    output_dtime : str
        The datetime string to use for the output filename
    Raises
    ------
    NotImplementedError
        _description_
    """
    import matlab.engine  # for saving data to workspace

    if dset_type in ["bilat", "chr2", "tmev"]:
        output_fpath = os.path.join(
            output_folder, f"loco_chr2_{output_dtime}.mat")
    else:
        raise NotImplementedError(
            f"save_to_workspace: Unknown dataset type: {dset_type}"
        )
    print(f"Saving session-level data to workspace\n\t{output_fpath}")
    eng = matlab.engine.start_matlab()
    for colname in df_to_save.columns:
        dtype = df_to_save[colname].dtype
        if "%" in colname:
            colname_matlab = colname.replace("%", "percent")
        else:
            colname_matlab = colname
        if dtype == np.object_:  # strings are represented as object_ in np array
            eng.workspace[colname_matlab] = list(np.array(df_to_save[colname]))
        elif dtype == np.int64:
            eng.workspace[colname_matlab] = matlab.int64(
                list(df_to_save[colname]))
        elif dtype == np.int32:
            eng.workspace[colname_matlab] = matlab.int32(
                list(df_to_save[colname]))
        elif dtype == np.float64:
            eng.workspace[colname_matlab] = matlab.double(
                list(df_to_save[colname]))
        else:
            raise NotImplementedError(f"{dtype} not implemented yet!")

    eng.eval(f"save('{output_fpath}')", nargout=0)
    print("Saved successfully.")
    eng.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fpath",
        type=str,
        default=None,
        help="Path to the assembled traces hdf5 file",
    )
    parser.add_argument(
        "--ampl_threshold",
        type=float,
        default=0.2,
        help="threshold that one element within the running episode candidate has to be reached for the episode to not be discarded",
    )
    parser.add_argument(
        "--temp_threshold",
        type=int,
        default=15,
        help="Threshold for duration that a candidate episode has to reach to not be discarded",
    )
    parser.add_argument(
        "--episode_merge_threshold",
        type=int,
        default=8,
        help="Merge running episodes if temporal distance distance smaller than this many frames or equal (15 Hz!)",
    )
    parser.add_argument(
        "--save_data", action="store_true", help="Save data to Excel file"
    )
    parser.add_argument("--save_figs", action="store_true",
                        help="Save figures")
    parser.add_argument(
        "--file_format", type=str, default="pdf", help="File format for figures"
    )
    parser.add_argument(
        "--save_sanity_check", action="store_true", help="Save sanity check figure"
    )
    parser.add_argument(
        "--save_waterfall", action="store_true", help="Save waterfall plot"
    )
    args = parser.parse_args()
    # TODO: check if this returns the tuple of dataframes in each use case. (Calling from command line, for example)
    # TODO: create Params object to pass to main function
    main(
        args.fpath,
        args.ampl_threshold,
        args.temp_threshold,
        args.episode_merge_threshold,
        args.save_data,
        args.save_figs,
        args.file_format,
        args.save_sanity_check,
        args.save_waterfall,
    )
