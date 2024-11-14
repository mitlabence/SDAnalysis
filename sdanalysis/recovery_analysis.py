"""
recovery_analysis.py - Recovery analysis pipeline for TMEV and window stimulation datasets.
"""

import os
from dataclasses import dataclass
from typing import Tuple
import warnings
from math import floor, ceil
from datetime import datetime
import seaborn as sns
import h5py
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial
import env_reader
import data_documentation as dd


@dataclass
class RecoveryAnalysisParams:
    """Data class containing parameters for the analysis.

    Attributes
    ----------
    percent_considered : float
        x% darkest/brightest of complete trace to consider
    extreme_group_size : int
        this many of the darkest/brightest pixels to consider
        (earliest darkest percent_considered% pixels)
    n_trough_frames : int
        Upper limit of window where to look for darkest point
    sd_window_width_frames : int
        length of window beginning with "post" (appearance of first SD wave visually) segment
        to look for amplitude of SD
    recovery_ratio : float
        The ratio (max: 1.0) of baseline to be reached to be considered recovered
    peak_window_length : int
        consider the first <peak_window_length> frames when looking for peak
    window_width_s : int
        width of the window in seconds
    window_step_s : int
        step of the window in seconds
    imaging_frequency : float
        imaging frequency in Hz
    n_frames_before_nc : int
        include frames just before manually detected "post-Sz" segment
    n_frames_before_ca1 : int
        number of frames before the manually detected "post-Sz" segment to include
    n_windows_post_darkest : int
        number of windows post darkest point to consider for recovery test.
    default_bl_center_ca1 : int
        baseline center should be this many frames before seizure/stim begin (CA1 window)
    default_bl_center_nc : int
        baseline center should be this many frames before seizure/stim begin (NC window)
    """

    # General parameters
    percent_considered: float = 5
    extreme_group_size: int = 15
    n_trough_frames: int = 5000
    sd_window_width_frames = 450
    recovery_ratio = 0.95
    # Time window-related parameters
    peak_window_length: int = 600
    window_width_s: int = 10
    window_step_s: int = 5
    imaging_frequency: float = 15.0
    n_frames_before_nc: int = 200
    n_frames_before_ca1: int = 0
    n_windows_post_darkest: int = 300
    default_bl_center_ca1: int = (
        -75
    )  # bl center this many frames before seizure/stim begin
    default_bl_center_nc: int = -975
    # manual corrections to the neocortical dataset (n_frames_before_nc)
    n_frames_before_post_start_nc = {
        "2251bba132cf45fa839d3214d1651392": 125,
        "4dea78a01bf5408092f498032d67d84e": 205,
        "54c31c3151944cfd86043932d3a19b9a": 60,
        "5cfb012d47f14303a40680d2b333336a": 125,
        "7753b03a2a554cccaab42f1c0458d742": 70,
        "cd3c1e0e3c284a89891d2e4d9a7461f4": 192,
        "f481149fa8694621be6116cb84ae2d3c": 115,
        "f5ccb81a34bb434482e2498bfdf88784": 58,
    }
    win_types_mapping = {"CA1": "CA1", "Cx": "NC"}  # replace Cx with NC

    # calculated quantities
    def window_width_frames(self) -> int:
        """getter for window width (calculated from window_width_s and imaging_frequency) in frames
        Returns:
            int: the selected window width in frames unit
        """
        return int(self.window_width_s * self.imaging_frequency)

    def window_step_frames(self) -> int:
        """getter for window step (calculated from window_step_s and imaging_frequency) in frames.
        Window step is the offset between two consecutive center coordinates of windows
        Returns:
            int: the window step size in frames unit
        """
        return int(self.window_step_s * self.imaging_frequency)

    def half_window_width_frames(self) -> int:
        """The half of the window width in frames. If window width is uneven,
        this is the floor of the division by 2.
        For example:
        If window width is 11, half window width is 5.
        If window width is 10, half window width is 5.
        Returns:
            int: Half window size in frames unit
        """
        return self.window_width_frames() // 2


@dataclass
class RecoveryAnalysisData:
    """Data class containing data for the recovery analysis.

    Attributes
    ----------
    dict_pre_fluo : dict
        {uuid: [fluorescence values]}
    dict_mid_fluo : dict
        {uuid: [fluorescence values]}
    dict_post_fluo : dict
        {uuid: [fluorescence values]}
    dict_meta : dict
        {uuid: {
            "exp_type": exp_type,
            "mouse_id": mouse_id,
            "session_uuids": [session_uuids],
            "segment_type_break_points": [segment_type_break_points]
        }}
    dict_segment_break_points : dict
        {uuid: (i_begin_mid, i_begin_am),
        pre: [:i_begin_mid],
        mid: [i_begin_mid:i_begin_am],
        post: [i_begin_am:]}
    dict_excluded : dict
        {uuid: {
            "exp_type": exp_type,
            "mouse_id": mouse_id,
            "win_type": window_type,
            "session_uuids": [session_uuids]
        }}
    """

    dict_pre_fluo: dict
    dict_mid_fluo: dict
    dict_post_fluo: dict
    dict_meta: dict
    dict_segment_break_points: dict
    dict_excluded: dict

    def __add__(self, other: "RecoveryAnalysisData") -> "RecoveryAnalysisData":
        return RecoveryAnalysisData(
            self.dict_pre_fluo.copy().update(other.dict_pre_fluo),
            self.dict_mid_fluo.copy().update(other.dict_mid_fluo),
            self.dict_post_fluo.copy().update(other.dict_post_fluo),
            self.dict_meta.copy().update(other.dict_meta),
            self.dict_segment_break_points.copy().update(
                other.dict_segment_break_points
            ),
            self.dict_excluded.copy().update(other.dict_excluded),
        )


def load_recovery_data(
    fpath: str, ddoc: dd.DataDocumentation, params: RecoveryAnalysisParams
) -> RecoveryAnalysisData:
    """Load a recovery hdf5 file

    Args:
    ----
        fpath (str): the path to the hdf5 file
        ddoc (DataDocumentation): the data documentation object
        params (RecoveryAnalysisParams): the recovery analysis parameters object

    Returns:
    -------
        RecoveryAnalysisData:
            The loaded data
    """
    dict_post_fluo = {}  # uuid: [mean_fluo], cut to post-segment (+ extra frames) only!
    dict_pre_fluo = {}  # baseline (until segment_type_break_points[1])
    dict_mid_fluo = {}  # rest of trace: sz or stim+sz
    # to get complete trace for event_uuid:
    # np.concatenate([dict_bl_fluo[event_uuid], dict_mid_fluo[event_uuid],
    # dict_mean_fluo[event_uuid]])
    dict_meta = {}
    dict_excluded = {}
    dict_segment_break_points = {}
    with h5py.File(fpath, "r") as h_f:
        for event_uuid, event_uuid_grp in h_f.items():
            win_type = params.win_types_mapping[event_uuid_grp.attrs["window_type"]]
            assert "session_uuids" in event_uuid_grp.attrs
            mouse_id = event_uuid_grp.attrs["mouse_id"]
            # for TMEV, traces were stitched together from multiple recordings, so uuid is not in
            # data documentation.
            # But the individual session uuids are stored in attributes (both for ChR2 and
            # TMEV data)
            session_uuids = event_uuid_grp.attrs["session_uuids"]
            exp_type = ddoc.getExperimentTypeForUuid(session_uuids[0])
            mean_fluo = np.array(event_uuid_grp["mean_fluo"])
            segment_type_break_points = event_uuid_grp.attrs[
                "segment_type_break_points"
            ]
            if exp_type == "tmev":
                # as TMEV traces are stitched together, it is difficult to use data documentation.
                # But segment_type_break_points attribute contains bl, sz, am begin frames.
                # am (aftermath) is defined as visual appearance of first SD wave. Can take this
                # as beginning
                assert (
                    len(segment_type_break_points) == 3
                )  # make sure only bl, sz, am points are in list
                i_begin_am = segment_type_break_points[2]
                i_begin_mid = segment_type_break_points[
                    1
                ]  # one frame past end of baseline, i.e. begin of middle section (sz)
                if (
                    win_type == "NC"
                ):  # NC seizures end abruptly, manual segmentation tries to set
                    # "reaching darkest point" as end of Sz. This means trough might be
                    # missed in original "aftermath" category.
                    i_begin_am -= params.n_frames_before_post_start_nc[event_uuid]
                    assert i_begin_am > 0
            elif exp_type in ["chr2_sd", "chr2_szsd"]:
                assert session_uuids[0] == event_uuid
                df_segments = ddoc.getSegmentsForUUID(event_uuid)
                # set first frame of first SD appearance as beginning
                i_begin_am = (
                    df_segments[
                        df_segments["interval_type"] == "sd_wave"
                    ].frame_begin.min()
                    - 1
                )  # 1-indexing to 0-indexing conversion
                i_begin_mid = (
                    df_segments[
                        df_segments["interval_type"] == "stimulation"
                    ].frame_begin.min()
                    - 1
                )
            else:
                continue  # do not add chr2_ctl recordings to dataset
            if not np.isnan(i_begin_am):
                bl_fluo = mean_fluo[:i_begin_mid].copy()
                mid_fluo = mean_fluo[i_begin_mid:i_begin_am].copy()
                if not len(mid_fluo) > 0:
                    print(f"{i_begin_mid} - {i_begin_am}")
                mean_fluo = mean_fluo[i_begin_am:]

                dict_segment_break_points[event_uuid] = (i_begin_mid, i_begin_am)

                dict_pre_fluo[event_uuid] = bl_fluo
                dict_post_fluo[event_uuid] = mean_fluo
                dict_mid_fluo[event_uuid] = mid_fluo
                dict_meta[event_uuid] = {
                    "exp_type": exp_type,
                    "mouse_id": mouse_id,
                    "win_type": win_type,
                    "session_uuids": session_uuids,
                    "segment_type_break_points": segment_type_break_points,
                }
            else:
                dict_excluded[event_uuid] = {
                    "exp_type": exp_type,
                    "mouse_id": mouse_id,
                    "win_type": win_type,
                    "session_uuids": session_uuids,
                    "segment_type_break_points": segment_type_break_points,
                }
    return RecoveryAnalysisData(
        dict_pre_fluo=dict_pre_fluo,
        dict_mid_fluo=dict_mid_fluo,
        dict_post_fluo=dict_post_fluo,
        dict_meta=dict_meta,
        dict_segment_break_points=dict_segment_break_points,
        dict_excluded=dict_excluded,
    )


def get_window(i_center, trace, params: RecoveryAnalysisParams) -> np.array:
    """Given i_center and the global parameter half_window_width_frames, try to return a window
    centered around i_center, and with inclusive borders at i_center - half_window_width_frames,
    i_center + half_window_width_frames. Might return a smaller window
    [0, i_center + half_window_width_frames], or
    [i_center - half_window_width_frames, len(trace) - 1]
    if the boundaries are outside the shape of trace.
    Parameters
    ----------
    i_center : int
        The index of center of the window in trace
    trace : np.array
        The trace to extract the window from
    params : RecoveryAnalysisParams
        The recovery analysis parameters object

    Returns
    -------
    np.array
        The window, a subarray of trace
    """
    if i_center > len(trace):
        warnings.warn(
            f"Trying to access window with center {i_center}, but only {len(trace)} frames"
        )
        return np.array([])
    if i_center + params.half_window_width_frames > len(trace):
        warnings.warn(
            f"Part of window out of bounds: {i_center} + HW \
                {params.half_window_width_frames} > {len(trace)}"
        )
        right_limit = len(trace)
    else:
        right_limit = (
            i_center + params.half_window_width_frames + 1
        )  # right limit is exclusive
    if i_center - params.half_window_width_frames < 0:
        warnings.warn(
            f"Part of window out of bounds: {i_center} - HW {params.half_window_width_frames} < 0"
        )
        left_limit = 0
    else:
        left_limit = i_center - params.half_window_width_frames
    return trace[left_limit:right_limit]


def get_window_for_event_type(
    event_uuid: str,
    data: RecoveryAnalysisData,
    params: RecoveryAnalysisParams,
    ddoc: dd.DataDocumentation,
    event_type="sz",
):
    """Given the event uuid and the event type (sz, sd) to look for, return np.array()
    of the corresponding window in
    the whole trace of the original hdf5 data

    Parameters
    ----------
    event_uuid : str
        the event_uuid of the trace (the name of the hdf5 group)
    data: RecoveryAnalysisData
        the opened data for recovery analysis (see load_recovery_data() function)
    params: RecoveryAnalysisParams
        the parameters for the recovery analysis
    ddoc: dd.DataDocumentation
        the data documentation object
    event_type : str
        "sd" or "sz". The event for which the window to be returned: end of bl/stim until beginning
        of first SD if "sz", else a fixed 30s window starting with the appearance of the first
        SD wave.
    Returns
    -------
    np.array
        The window (empty array if event_type does not exist for the recording type)
    """
    exp_type = data.dict_meta[event_uuid]["exp_type"]
    win_type = data.dict_meta[event_uuid]["win_type"]
    complete_trace = np.concatenate(
        [
            data.dict_pre_fluo[event_uuid],
            data.dict_mid_fluo[event_uuid],
            data.dict_post_fluo[event_uuid],
        ]
    )

    if exp_type == "tmev":
        break_points = data.dict_meta[event_uuid][
            "segment_type_break_points"
        ]  # [bl_begin, sz_begin, SD_begin]
        if (
            event_type == "sz"
        ):  # am begins with appearance of first SD wave -> if sz, get time second and third indices
            return complete_trace[break_points[1] : break_points[2]]
        if (
            event_type == "sd" and win_type != "NC"
        ):  # prove me wrong, but no SD in NC. :)
            return complete_trace[
                break_points[2] : break_points[2] + params.sd_window_width_frames
            ]
    elif "chr2" in exp_type:
        df_segments = ddoc.getSegmentsForUUID(
            event_uuid
        )  # sessions consist of one recording, so event_uuid = recording_uuid
        if event_type == "sz" and exp_type == "chr2_szsd":
            i_begin_sz = (
                df_segments[df_segments["interval_type"] == "sz"].frame_begin.iloc[0]
                - 1
            )  # switch to 0-based indexing
            i_end_sz = df_segments[df_segments["interval_type"] == "sz"].frame_end.iloc[
                0
            ]  # upper limit exclusive
            return complete_trace[i_begin_sz:i_end_sz]
        if event_type == "sd" and "sd" in exp_type:
            i_begin_sd = (
                df_segments[df_segments["interval_type"] == "sd_wave"].frame_begin.iloc[
                    0
                ]
                - 1
            )  # switch to 0-based indexing
            i_end_sd = i_begin_sd + params.sd_window_width_frames
            return complete_trace[i_begin_sd:i_end_sd]
    warnings.warn("No window found!")
    return np.array([])


def get_metric_for_window(trace_window: np.array, params: RecoveryAnalysisParams):
    """Given a window, calculate the following metric:
    1. Take percent_considered % of the lowest values within the window
    2. Get the median value of the values found in step 1.

    Parameters
    ----------
    trace_window : np.array
        The window to calculate the metric for.
    params: RecoveryAnalysisParams
        The parameters for the recovery analysis

    Returns
    -------
    float
        The calculated metric
    """
    lowest_indices = np.argsort(trace_window)[
        : int(params.percent_considered / 100.0 * len(trace_window))
    ]
    lowest_values = trace_window[lowest_indices]
    return np.median(lowest_values)


def get_peak_metric(trace_window: np.array):
    """Given a trace window, calculate the mean of top 5% values.
        Intended use: SD and Sz amplitudes.
    Parameters
    ----------
    trace_window : np.array
        The window to calculate the metric for.

    Returns
    -------
    float
        The calculated metric
    """

    mean_top_5p = np.flip(np.sort(trace_window))[
        : int(0.05 * len(trace_window))
    ].mean()  # take mean of highest 5% of sz values
    return mean_top_5p


def main(
    fpath_stim_dset: str,
    fpath_tmev_dset: str,
    save_results: bool = False,
    params: RecoveryAnalysisParams = RecoveryAnalysisParams(),
):
    """Perform recovery analysis.

    Args:
        fpath_stim_dset (str): File path of the stimulation dataset (hdf5 file)
        fpath_tmev_dset (str): File path of the TMEV dataset (hdf5 file)
        save_results (bool, optional): Whether to save results. Defaults to False.
        params (RecoveryAnalysisParams, optional): The analysis parameters.
        Defaults to RecoveryAnalysisParams().

    Raises:
        FileNotFoundError: _description_
        FileNotFoundError: _description_
        KeyError: _description_
    """
    if fpath_stim_dset is None or not os.path.exists(fpath_stim_dset):
        raise FileNotFoundError(f"Stim dataset file not found at\n\t{fpath_stim_dset}")
    if fpath_tmev_dset is None or not os.path.exists(fpath_tmev_dset):
        raise FileNotFoundError(f"TMEV dataset file not found at\n\t{fpath_tmev_dset}")
    env_dict = env_reader.read_env()
    data_doc = dd.DataDocumentation(env_dict["DATA_DOCU_FOLDER"])
    if "OUTPUT_FOLDER" not in env_dict:
        raise KeyError("OUTPUT_FOLDER not found in the environment file.")
    output_folder = env_dict["OUTPUT_FOLDER"]
    ddoc = dd.DataDocumentation.from_env_dict(env_dict)
    dataset = load_recovery_data(fpath_stim_dset, ddoc, params)
    for fpath_dset in [fpath_tmev_dset]:  # add more datasets here if needed
        dataset_temp = load_recovery_data(fpath_dset, ddoc, params)
        dataset += dataset_temp
