"""
recovery_analysis.py - Recovery analysis pipeline for TMEV and window stimulation datasets.
"""

import argparse
import os
from dataclasses import dataclass
import warnings
from math import floor, ceil
import h5py
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial
import env_reader
import data_documentation as dd
from custom_io import open_file, get_datetime_for_fname


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
    manual_bl_centers : dict
        manual corrections of baseline frame indices to the neocortical dataset
        (default_bl_center_nc/ca1)
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
    manual_bl_centers = {
        "aa66ae0470a14eb08e9bcadedc34ef64": -750,
        "c7b29d28248e493eab02288b85e3adee": -1000,
        "7b9c17d8a1b0416daf65621680848b6a": -950,
        "9e75d7135137444492d104c461ddcaac": -300,
        "d158cd12ad77489a827dab1173a933f9": -500,
        "a39ed3a880c54f798eff250911f1c92f": -500,
        "4e2310d2dde845b0908519b7196080e8": -500,
        "f0442bebcd1a4291a8d0559eb47df08e": -500,
        "2251bba132cf45fa839d3214d1651392": -1300,
        "cd3c1e0e3c284a89891d2e4d9a7461f4": -1500,
    }

    win_types_mapping = {"CA1": "CA1", "Cx": "NC"}  # replace Cx with NC

    # calculated quantities
    @property
    def window_width_frames(self) -> int:
        """getter for window width (calculated from window_width_s and imaging_frequency) in frames
        Returns:
            int: the selected window width in frames unit
        """
        return int(self.window_width_s * self.imaging_frequency)

    @property
    def window_step_frames(self) -> int:
        """getter for window step (calculated from window_step_s and imaging_frequency) in frames.
        Window step is the offset between two consecutive center coordinates of windows
        Returns:
            int: the window step size in frames unit
        """
        return int(self.window_step_s * self.imaging_frequency)

    @property
    def half_window_width_frames(self) -> int:
        """The half of the window width in frames. If window width is uneven,
        this is the floor of the division by 2.
        For example:
        If window width is 11, half window width is 5.
        If window width is 10, half window width is 5.
        Returns:
            int: Half window size in frames unit
        """
        return self.window_width_frames // 2


@dataclass
class RecoveryAnalysisData:
    """Data class containing data for the recovery analysis.

    Attributes
    ----------
    dict_bl_fluo : dict
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

    dict_bl_fluo: dict
    dict_mid_fluo: dict
    dict_post_fluo: dict
    dict_meta: dict
    dict_segment_break_points: dict
    dict_excluded: dict

    def copy(self) -> "RecoveryAnalysisData":
        """Create a deep copy of the RecoveryAnalysisData object

        Returns:
            RecoveryAnalysisData: the copied object
        """
        return RecoveryAnalysisData(
            dict_bl_fluo=self.dict_bl_fluo.copy(),
            dict_mid_fluo=self.dict_mid_fluo.copy(),
            dict_post_fluo=self.dict_post_fluo.copy(),
            dict_meta=self.dict_meta.copy(),
            dict_segment_break_points=self.dict_segment_break_points.copy(),
            dict_excluded=self.dict_excluded.copy(),
        )

    def __add__(self, other: "RecoveryAnalysisData") -> "RecoveryAnalysisData":
        dict_bl_fluo_result = self.dict_bl_fluo.copy()
        dict_bl_fluo_result.update(other.dict_bl_fluo)
        dict_mid_fluo_result = self.dict_mid_fluo.copy()
        dict_mid_fluo_result.update(other.dict_mid_fluo)
        dict_post_fluo_result = self.dict_post_fluo.copy()
        dict_post_fluo_result.update(other.dict_post_fluo)
        dict_meta_result = self.dict_meta.copy()
        dict_meta_result.update(other.dict_meta)
        dict_segment_break_points_result = self.dict_segment_break_points.copy()
        dict_segment_break_points_result.update(other.dict_segment_break_points)
        dict_excluded_result = self.dict_excluded.copy()
        dict_excluded_result.update(other.dict_excluded)
        return RecoveryAnalysisData(
            dict_bl_fluo_result,
            dict_mid_fluo_result,
            dict_post_fluo_result,
            dict_meta_result,
            dict_segment_break_points_result,
            dict_excluded_result,
        )

    def __eq__(self, other: "RecoveryAnalysisData") -> bool:
        return (
            self.dict_bl_fluo == other.dict_bl_fluo
            and self.dict_mid_fluo == other.dict_mid_fluo
            and self.dict_post_fluo == other.dict_post_fluo
            and self.dict_meta == other.dict_meta
            and self.dict_segment_break_points == other.dict_segment_break_points
            and self.dict_excluded == other.dict_excluded
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
    dict_bl_fluo = {}  # baseline (until segment_type_break_points[1])
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
            exp_type = ddoc.get_experiment_type_for_uuid(session_uuids[0])
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
                df_segments = ddoc.get_segments_for_uuid(event_uuid)
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

                dict_bl_fluo[event_uuid] = bl_fluo
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
        dict_bl_fluo=dict_bl_fluo,
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
            data.dict_bl_fluo[event_uuid],
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
        ):  # am begins with appearance of first SD wave -> if sz, get time second and third
            # indices
            return complete_trace[break_points[1] : break_points[2]]
        if (
            event_type == "sd" and win_type != "NC"
        ):  # prove me wrong, but no SD in NC. :)
            return complete_trace[
                break_points[2] : break_points[2] + params.sd_window_width_frames
            ]
    elif "chr2" in exp_type:
        df_segments = ddoc.get_segments_for_uuid(
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


def get_peak_metric(trace_window: np.array) -> float:
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


def get_bl_window_metric(
    analysis_data: RecoveryAnalysisData, analysis_params: RecoveryAnalysisParams
) -> dict:
    """Given the analysis data and parameters, calculate the baseline window metric for each

    Args:
        analysis_data (RecoveryAnalysisData): _description_
        analysis_params (RecoveryAnalysisParams): _description_

    Returns:
        dict: uuid: (i_bl, bl_metric), i_bl is the center of the window, bl_metric is the metric
    """
    # uuid: (i_bl, bl_metric), i_bl is the center of the window
    dict_bl_values = {}

    for (
        uuid
    ) in (
        analysis_data.dict_meta.keys()
    ):  # uuid: {"exp_type": exp_type, "mouse_id": mouse_id, "session_uuids": [session_uuids]}
        exp_type = analysis_data.dict_meta[uuid]["exp_type"]
        win_type = analysis_data.dict_meta[uuid]["win_type"]
        # check if manually corrected. If not, check if TMEV or not. If TMEV,
        # use default_bl_center_ca1/default_bl_center_nc
        # if ChR2, can use a window right before stim
        bl_trace = analysis_data.dict_bl_fluo[uuid]
        if uuid in analysis_params.manual_bl_centers:
            i_bl = analysis_params.manual_bl_centers[uuid]
        elif exp_type == "tmev":
            if win_type == "CA1":
                i_bl = analysis_params.default_bl_center_ca1
            elif win_type == "NC":
                i_bl = analysis_params.default_bl_center_nc
        elif exp_type in ["chr2_sd", "chr2_szsd"]:
            # take a window just before stim
            i_bl = len(bl_trace) - analysis_params.half_window_width_frames - 1
        if i_bl < 0:
            i_bl = len(bl_trace) + i_bl
        bl_win = get_window(i_bl, bl_trace, analysis_params)
        bl_metric = get_metric_for_window(bl_win, analysis_params)
        dict_bl_values[uuid] = (i_bl, bl_metric)
    return dict_bl_values


def get_significant_time_points(
    analysis_data: RecoveryAnalysisData, analysis_params: RecoveryAnalysisParams
) -> dict:
    """
    Given the analysis data and parameters, calculate the significant time points for each
    event

    Args:
        analysis_data (RecoveryAnalysisData): _description_
        analysis_params (RecoveryAnalysisParams): _description_

    Returns:
        dict: (uuid:
            (i_sd_peak, i_trough, i_fwhm, peak_amplitude, trough_amplitude, fwhm_amplitude))
        i_sd_peak: index of SD peak
        i_trough: index of trough
        i_fwhm: index of half maximum of SD peak (after SD peak)
        peak_amplitude: amplitude of SD peak
        trough_amplitude: amplitude (metric) of trough
        fwhm_amplitude: amplitude of half maximum of SD peak
    """
    # aftermath:
    # TMEV - appearance of first SD. This could also be taken above
    # ChR2 - if SD present, then appearance of first SD. Else: directly after stim (ctl).

    dict_significant_tpoints = {}  # uuid: (i_sd_peak, i_trough, i_fwhm, peak_amplitude,
    #   trough_amplitude, fwhm_amplitude=peak_amplitude/2)

    for event_uuid, post_trace in analysis_data.dict_post_fluo.items():
        win_type = analysis_data.dict_meta[event_uuid]["win_type"]
        # traces already cut to "aftermath" (plus few extra frames)

        # get 5% darkest points of "post" segment
        sorted_beginning = np.argsort(post_trace[: analysis_params.peak_window_length])
        i_brightest = sorted_beginning[
            -1
        ]  # this is supposed to be SD amplitude. Later, set it to np.nan if there is no SD.
        i_sd_peak = i_brightest
        cut_trace = post_trace[
            i_brightest : i_brightest + analysis_params.n_trough_frames
        ]
        # use reduced window to look for trough
        sorted_indices_cut = np.argsort(cut_trace)
        i_darkest_group = sorted_indices_cut[
            : int(analysis_params.percent_considered / 100.0 * len(post_trace))
        ]  # still take n percent of aftermath, not cut trace!
        # get single coordinate for darkest part
        # find darkest <percent_considered>%, take earliest <extreme_group_size> of them, get
        # median frame index of these, round down to integer frame
        i_darkest_cut = int(
            floor(
                np.median(
                    np.sort(i_darkest_group)[: analysis_params.extreme_group_size]
                )
            )
        )
        # i_darkest_cut = sorted_indices_cut[0]  # Find absolute minimum value

        i_darkest = (
            i_darkest_cut + i_brightest
        )  # bring it back to original frame indices

        # get Sz and SD amplitude metrics
        # y_brightest = complete_trace[i_brightest]
        # TODO: originally, i_brightest was the index of maximum brightness. In Baseline recovery,
        # the SD amplitude uses different approach than implemented below. Need to remove
        # i_brightest and old y_brightest = complete_trace[i_brightest]! (brightest is SD peak)
        # sd_window = get_window_for_event_type(event_uuid, "sd")
        # y_sd_peak = get_peak_metric(sd_window)
        # if len(sd_window) == 0:
        #    i_sd_peak = np.nan
        if (
            win_type == "CA1"
        ):  # TODO: Assume no SD in NC window. Change this to data documentation based decision!
            y_sd_peak = post_trace[i_sd_peak]
        else:
            y_sd_peak = np.nan

        # sz_window = get_window_for_event_type(event_uuid, "sz")
        # y_sz_peak = get_peak_metric(sz_window)

        # y_darkest = complete_trace[i_darkest]  # TODO: get window value instead?
        y_darkest = get_window(i_darkest, post_trace, analysis_params)
        y_darkest = get_metric_for_window(y_darkest, analysis_params)

        # find time of half maximum
        if not np.isnan(i_sd_peak):
            y_half = (y_sd_peak + y_darkest) / 2.0  # bl + (peak - bl)/2
            i_half = np.argmax(post_trace[i_brightest:] <= y_half)
            i_half += i_sd_peak
        else:
            i_half = np.nan
            y_half = np.nan
        # print()
        # print(i_darkest)
        # print(i_half)
        # assert i_brightest < i_half
        # assert i_darkest > i_half
        if (
            win_type == "NC"
        ):  # no SD in NC windows... Correct this logic if I'm wrong :)
            print(f"{event_uuid} window type is NC. No SD = no peak.")
            i_sd_peak = np.nan
            i_half = np.nan
            y_sd_peak = np.nan
            y_half = np.nan
        dict_significant_tpoints[event_uuid] = (
            i_sd_peak,
            i_darkest,
            i_half,
            y_sd_peak,
            y_darkest,
            y_half,
        )
    return dict_significant_tpoints


def get_seizure_amplitude(
    analysis_data: RecoveryAnalysisData,
    ddoc: dd.DataDocumentation,
) -> dict:
    """Given the analysis data and parameters, calculate the seizure amplitude (if present)
    for each uuid

    Args:
        analysis_data (RecoveryAnalysisData): _description_
        ddoc (dd.DataDocumentation): _description_

    Returns:
        dict: _description_
    """
    # uuid: (i_mid_max, y_mid_max) where i_mid_max is the frame index of the
    #  mid segment (dict_mid_fluo). Stim is ignored when finding the max.
    dict_sz_amps = (
        {}
    )  # contains y_sz_max, i.e. absolute amplitude, not compared to baseline!

    for event_uuid in analysis_data.dict_post_fluo.keys():
        exp_type = analysis_data.dict_meta[event_uuid]["exp_type"]
        if exp_type in [
            "chr2_szsd",
            "tmev",
        ]:  # only consider recordings where seizure occurs
            mid_trace = analysis_data.dict_mid_fluo[event_uuid]
            if exp_type == "chr2_szsd":  # ignore stim frames
                df_segments = ddoc.get_segments_for_uuid(event_uuid)
                assert (
                    "sz" in df_segments.interval_type.unique()
                )  # make sure sz actually occurred
                # get number of stim frames to ignore. mid section begins with stim frames.
                i_begin_stim = df_segments[
                    df_segments["interval_type"] == "stimulation"
                ].frame_begin.iloc[
                    0
                ]  # inclusive
                i_end_stim = df_segments[
                    df_segments["interval_type"] == "stimulation"
                ].frame_end.iloc[
                    0
                ]  # inclusive
                n_stim_frames = i_end_stim - i_begin_stim + 1
                mid_trace = mid_trace[n_stim_frames:]
            else:  # if tmev, make sure sz segment exists
                session_uuids = analysis_data.dict_meta[event_uuid]["session_uuids"]
                sz_present = False
                for session_uuid in session_uuids:
                    df_segments = ddoc.get_segments_for_uuid(session_uuid)
                    if "sz" in df_segments.interval_type.unique():
                        sz_present = True
                        break
                if not sz_present:
                    dict_sz_amps[event_uuid] = np.nan
                    print(event_uuid)
                    continue
            dict_sz_amps[event_uuid] = np.max(mid_trace)
        else:  # no seizure in experiment
            dict_sz_amps[event_uuid] = np.nan
    return dict_sz_amps


def get_windows_from(i_begin_center, trace, analysis_params: RecoveryAnalysisParams):
    """Given a trace and the 0-based index of the center of a first window, return the indices
    and the corresponding window metrics.

    Parameters
    ----------
    i_begin_center : int
        The 0-based index of the first window center to include
    trace : np.array
        The trace
    analysis_params (RecoveryAnalysisParams): the analysis parameters object

    Returns
    -------
    tuple(np.array, np.array)
        A tuple with two arrays: at location 0, the 0-based window centers and at location 1,
        the corresponding window metrics.
    """
    i_center_current = i_begin_center
    x_vals = []
    y_vals = []
    while (
        i_center_current < len(trace) - analysis_params.half_window_width_frames
    ):  # stop algorithm upon reaching end of recording
        current_win = get_window(i_center_current, trace, analysis_params)
        y_current = get_metric_for_window(current_win, analysis_params)
        x_vals.append(i_center_current)
        y_vals.append(y_current)

        i_center_current += analysis_params.window_step_frames
    return (np.array(x_vals), np.array(y_vals))


def get_windows(
    dict_post_fluo: dict,
    dict_significant_tpoints: dict,
    dict_bl_window_metrics: dict,
    analysis_params: RecoveryAnalysisParams,
) -> dict:
    """Given the analysis data and parameters, calculate the windows for each event

    Args:
        dict_post_fluo (dict): The uuid: trace dictionary of the post-segment traces
        dict_significant_tpoints (dict): The significant time points for each event
        dict_bl_window_metrics (dict): The baseline window metrics for each event
        analysis_params (RecoveryAnalysisParams): The analysis parameters object
    Returns:
        dict: event_uuid:
            [y_bl_window, y_darkest_window, y_post_darkest1, y_post_darkest2, ...y_recovery_window]
    """
    dict_windows = {}  #
    for event_uuid in dict_post_fluo:
        post_trace = dict_post_fluo[event_uuid]
        i_trough = dict_significant_tpoints[event_uuid][1]

        x_windows = (
            []
        )  # the 0-indexed window center coordinates [x_bl, x_trough, x_window1, ...]
        y_windows = (
            []
        )  # the corresponding window metrics [y_bl, y_trough, y_window1, ...]

        # add y_bl
        i_bl = dict_bl_window_metrics[event_uuid][0]
        y_bl = dict_bl_window_metrics[event_uuid][1]
        x_windows.append(i_bl)
        y_windows.append(y_bl)

        # add trough window to windows list
        i_current = i_trough
        current_win = get_window(
            i_current, post_trace, analysis_params
        )  # start with metric at trough
        y_current = get_metric_for_window(current_win, analysis_params)
        y_windows.append(y_current)
        x_windows.append(i_current)

        # move on to next window just after trough to start looking for recovery
        # (FIXME: in some cases, already trough is > 95% of bl! by definition we
        # demand recovery to happen after the trough?)
        i_current += analysis_params.window_step_frames
        current_win = get_window(i_current, post_trace, analysis_params)
        while (
            len(current_win) >= analysis_params.window_width_frames
        ):  # stop algorithm upon reaching end of recording
            y_current = get_metric_for_window(current_win, analysis_params)
            y_windows.append(y_current)
            x_windows.append(i_current)
            if y_current >= analysis_params.recovery_ratio * y_bl:  # recovery reached
                break
            # move to next window
            i_current += analysis_params.window_step_frames
            current_win = get_window(i_current, post_trace, analysis_params)
        dict_windows[event_uuid] = y_windows
    return dict_windows


def get_recovery(
    dict_post_fluo: dict,
    dict_significant_tpoints: dict,
    dict_bl_window_metrics: dict,
    analysis_params: RecoveryAnalysisParams,
) -> dict:
    """Given the analysis data and parameters, calculate the recovery time point for each event

    Args:
        dict_post_fluo (dict): _description_
        dict_significant_tpoints (dict): _description_
        dict_bl_window_metrics (dict): _description_
        analysis_params (RecoveryAnalysisParams): _description_

    Returns:
        dict: uuid: (x_recovery, y_recovery, did_recover)
        did_recover: boolean, whether recovery was found within the dataset
            If false, recovery was determined with extrapolation
    """
    dict_recovery = {}
    for event_uuid in dict_post_fluo:
        post_trace = dict_post_fluo[event_uuid]
        i_trough = dict_significant_tpoints[event_uuid][1]

        x_windows = (
            []
        )  # the 0-indexed window center coordinates [x_bl, x_trough, x_window1, ...]
        y_windows = (
            []
        )  # the corresponding window metrics [y_bl, y_trough, y_window1, ...]

        did_recover = False  # assume recovery will be found
        # add y_bl
        i_bl = dict_bl_window_metrics[event_uuid][0]
        y_bl = dict_bl_window_metrics[event_uuid][1]
        x_windows.append(i_bl)
        y_windows.append(y_bl)

        # add trough window to windows list
        i_current = i_trough
        current_win = get_window(
            i_current, post_trace, analysis_params
        )  # start with metric at trough
        y_current = get_metric_for_window(current_win, analysis_params)
        y_windows.append(y_current)
        x_windows.append(i_current)

        # move on to next window just after trough to start looking for recovery
        # (FIXME: in some cases, already trough is > 95% of bl! by definition we demand
        # recovery to happen after the trough?)
        i_current += analysis_params.window_step_frames
        current_win = get_window(i_current, post_trace, analysis_params)
        while (
            len(current_win) >= analysis_params.window_width_frames
        ):  # stop algorithm upon reaching end of recording
            y_current = get_metric_for_window(current_win, analysis_params)
            y_windows.append(y_current)
            x_windows.append(i_current)
            if y_current >= analysis_params.recovery_ratio * y_bl:  # recovery reached
                did_recover = True
                break
            # move to next window
            i_current += analysis_params.window_step_frames
            current_win = get_window(i_current, post_trace, analysis_params)
        # if no recovery found within trace, try to extrapolate. Start with window after trough.
        if not did_recover:
            y_recovery = analysis_params.recovery_ratio * y_bl
            x_recovery = try_extrapolate_recovery(
                x_windows[1:],
                y_windows[1:],
                y_recovery,
                analysis_params,
            )

        else:
            assert i_current < len(post_trace)
            x_recovery = i_current
            y_recovery = y_current

        dict_recovery[event_uuid] = (x_recovery, y_recovery, did_recover)
    return dict_recovery


def try_extrapolate_recovery(
    x_vals, y_vals, y_expol, analysis_params: RecoveryAnalysisParams
) -> int:
    """Given the x values x_vals and the corresponding y-values y_vals, try to find the x value
    corresponding to y_expol, based on linear extrapolation. This algorithm is specialized on
    finding recovery time point, so a line with positive slope is sought.
    If this cannot be found, a large time point is returned.
    Parameters
    ----------
    x_vals : np.array
        The x values (in the notebook, intended use is frames inb 0-based indexing, the center
        of windows)
    y_vals : np.array
        The y values (intended use is the metrics of the windows specified by x_vals)
    y_expol : int (or scalar, same as y_vals.dtype)
        The y value to extrapolate to. (intended use case: recovery_ratio*y_baseline)
    analysis_params: RecoveryAnalysisParams
        The analysis parameters object
    Returns
    -------
    int
        The found extrapolated time point, in same units as x_vals. (intended use case: the frame
        index where 95% of baseline is reached)
    """
    try:
        line_fit_coeffs = (
            Polynomial.fit([x_vals[0], x_vals[-1]], [y_vals[0], y_vals[-1]], deg=1)
            .convert()
            .coef
        )  # linear fit starting with point after darkest time point
        # the coefficients [a, b] from y= a + b*x.
        if (
            line_fit_coeffs[1] <= 0
        ):  # Check if b is non-positive -> No recovery possible
            # x_recovery_single_ca1.append(np.inf)
            # warnings.warn("Linear fit has negative slope!")
            print("Linear fit negative slope!")
            return (
                np.nan
            )  # set a very late recovery. TODO: come up with better value! np.inf messes up
        # statistics...
    except np.linalg.LinAlgError:
        print(
            "Could not extrapolate. Returning last window center as extrapolation time point..."
        )
        return x_vals[-1]
    # b>0 -> line is ascending, i.e. there will be a recovery time
    # find inverse function. We know y = a + b*x, need to have x = c + d*y,
    # where y = <threshold>*baseline (threshold=0.95)
    # inverse is x = -a/b + (1/b)*y = a_inv + b_inv*y
    a_inv = -line_fit_coeffs[0] / line_fit_coeffs[1]
    b_inv = 1 / line_fit_coeffs[1]
    x_recovery = a_inv + b_inv * analysis_params.recovery_ratio * y_expol
    x_recovery = ceil(x_recovery)
    if (
        x_recovery >= x_vals[-1]
    ):  # linear fit would cause extrapolated recovery to be earlier than last
        # non-recovered point
        x_recovery = x_vals[-1] + 2 * (
            x_vals[-1] - x_vals[-2]
        )  # the next window after the last is set as recovery
    return x_recovery


def create_dataframe(
    dict_significant_tpoints: dict,
    dict_bl_values: dict,
    dict_recovery: dict,
    dict_sz_amps: dict,
    analysis_data: RecoveryAnalysisData,
    analysis_params: RecoveryAnalysisParams,
) -> pd.DataFrame:
    """
    Given the results of bl extraction, rercovery and sz amplitude analysis, create a pandas
    dataframe with the results.

    Args:
        dict_significant_tpoints (dict): _description_
        dict_bl_values (dict): _description_
        dict_recovery (dict): _description_
        dict_sz_amps (dict): _description_
        analysis_data (RecoveryAnalysisData): _description_
        analysis_params (RecoveryAnalysisParams): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # (raw) columns: event_uuid, mouse_id, experiment_type, peak_time,
    # trough_time, peak_amplitude, trough_amplitude
    imaging_freq = analysis_params.imaging_frequency
    df_recovery = pd.DataFrame.from_dict(
        dict_significant_tpoints,
        "index",
        columns=["i_peak", "i_trough", "i_half", "y_peak", "y_trough", "y_half"],
    ).reset_index()
    # replace column name "index" with "event_uuid"
    df_recovery["event_uuid"] = df_recovery["index"]
    df_recovery = df_recovery.drop(columns=["index"])
    df_recovery["exp_type"] = df_recovery.apply(
        lambda row: analysis_data.dict_meta[row.event_uuid]["exp_type"], axis=1
    )
    df_recovery["mouse_id"] = df_recovery.apply(
        lambda row: analysis_data.dict_meta[row.event_uuid]["mouse_id"], axis=1
    )
    df_recovery["win_type"] = df_recovery.apply(
        lambda row: analysis_data.dict_meta[row.event_uuid]["win_type"], axis=1
    )

    df_recovery["y_bl"] = df_recovery.apply(
        lambda row: dict_bl_values[row["event_uuid"]][1], axis=1
    )
    df_recovery["i_bl"] = df_recovery.apply(
        lambda row: dict_bl_values[row["event_uuid"]][0], axis=1
    )

    # peak minus trough difference in amplitude
    df_recovery["dy_bl_trough"] = df_recovery["y_bl"] - df_recovery["y_trough"]
    # peak-trough time difference, s
    df_recovery["dt_peak_trough"] = (
        df_recovery["i_trough"] / imaging_freq - df_recovery["i_peak"] / imaging_freq
    )
    # peak to half amplitude time difference, s
    df_recovery["dt_peak_FWHM"] = (
        df_recovery["i_half"] / imaging_freq - df_recovery["i_peak"] / imaging_freq
    )

    df_recovery["i_recovery"] = df_recovery.apply(
        lambda row: dict_recovery[row["event_uuid"]][0], axis=1
    )
    df_recovery["y_recovery"] = df_recovery.apply(
        lambda row: dict_recovery[row["event_uuid"]][1], axis=1
    )
    df_recovery["did_recover"] = df_recovery.apply(
        lambda row: dict_recovery[row["event_uuid"]][2], axis=1
    )
    df_recovery["extrapolated"] = ~df_recovery["did_recover"]

    df_recovery["dt_trough_recovery"] = (
        df_recovery["i_recovery"] / imaging_freq
        - df_recovery["i_trough"] / imaging_freq
    )
    df_recovery["dt_peak_recovery"] = (
        df_recovery["i_recovery"] / imaging_freq - df_recovery["i_peak"] / imaging_freq
    )

    # move i_xy to whole trace indexing frame of reference
    df_recovery["i_recovery_whole"] = df_recovery.apply(
        lambda row: row["i_recovery"]
        + analysis_data.dict_segment_break_points[row["event_uuid"]][1],
        axis=1,
    )
    df_recovery["i_peak_whole"] = df_recovery.apply(
        lambda row: row["i_peak"]
        + analysis_data.dict_segment_break_points[row["event_uuid"]][1],
        axis=1,
    )
    df_recovery["i_trough_whole"] = df_recovery.apply(
        lambda row: row["i_trough"]
        + analysis_data.dict_segment_break_points[row["event_uuid"]][1],
        axis=1,
    )
    df_recovery["i_fwhm_whole"] = df_recovery.apply(
        lambda row: row["i_half"]
        + analysis_data.dict_segment_break_points[row["event_uuid"]][1],
        axis=1,
    )

    df_recovery["y_sz_max"] = df_recovery.apply(
        lambda row: dict_sz_amps[row["event_uuid"]], axis=1
    )

    df_recovery["dy_bl_sz"] = df_recovery["y_sz_max"] - df_recovery["y_bl"]
    df_recovery["dy_bl_sd"] = (
        df_recovery["y_peak"] - df_recovery["y_bl"]
    )  # peak of aftermath is the largest SD amplitude
    # final columns: event_uuid, mouse_id, exp_type, y_bl, y_peak, y_trough, y_recovery,
    # dy_trough_peak, dt_peak_trough, dt_peak_trough_FWHM, dt_trough_recovery, dt_peak_recovery,
    # did_recover
    df_recovery = df_recovery[
        [
            "event_uuid",
            "mouse_id",
            "exp_type",
            "win_type",
            "y_bl",
            "y_sz_max",
            "y_peak",
            "y_trough",
            "y_recovery",
            "dy_bl_sz",
            "dy_bl_sd",
            "dy_bl_trough",
            "dt_peak_trough",
            "dt_peak_FWHM",
            "dt_trough_recovery",
            "dt_peak_recovery",
            "extrapolated",
            "i_peak_whole",
            "i_fwhm_whole",
            "i_trough_whole",
            "i_recovery_whole",
        ]
    ].sort_values(by=["exp_type", "win_type", "event_uuid"])
    return df_recovery


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
    if "OUTPUT_FOLDER" not in env_dict:
        raise KeyError("OUTPUT_FOLDER not found in the environment file.")
    output_folder = env_dict["OUTPUT_FOLDER"]
    dtime_str = get_datetime_for_fname()
    ddoc = dd.DataDocumentation.from_env_dict(env_dict)
    dataset = load_recovery_data(fpath_stim_dset, ddoc, params)
    for fpath_dset in [fpath_tmev_dset]:  # add more datasets here if needed
        dataset_temp = load_recovery_data(fpath_dset, ddoc, params)
        dataset += dataset_temp
    dict_bl_values = get_bl_window_metric(dataset, params)
    dict_significant_tpoints = get_significant_time_points(dataset, params)
    dict_sz_amplitudes = get_seizure_amplitude(dataset, ddoc)
    dict_recovery = get_recovery(
        dataset.dict_post_fluo, dict_significant_tpoints, dict_bl_values, params
    )
    df_results = create_dataframe(
        dict_significant_tpoints,
        dict_bl_values,
        dict_recovery,
        dict_sz_amplitudes,
        dataset,
        params,
    )
    if save_results:
        # 1. Save recovery time results
        # the following dataframe contains recovery time and experiment metadata:
        # if did_recover is false, extrapolation was used
        df_recovery_time = df_results[
            [
                "mouse_id",
                "exp_type",
                "win_type",
                "event_uuid",
                "dt_trough_recovery",
                "extrapolated",
            ]
        ].sort_values(by=["exp_type", "win_type", "mouse_id"])
        df_recovery_time = df_recovery_time.rename(
            columns={"win_type": "window_type", "dt_trough_recovery": "t_recovery_s"}
        )
        fpath_recovery = os.path.join(output_folder, f"recovery_times_{dtime_str}.xlsx")
        df_recovery_time.to_excel(fpath_recovery, index=False)
        print(f"Saved recovery times to {fpath_recovery}")
        # 2. Save Bl-Sz, Bl-SD amplitudes
        df_amplitudes = df_results[
            ["mouse_id", "event_uuid", "exp_type", "win_type", "dy_bl_sz", "dy_bl_sd"]
        ].sort_values(by=["exp_type", "win_type", "mouse_id"])
        df_amplitudes = df_amplitudes.rename(
            columns={"dy_bl_sz": "Sz-bl", "dy_bl_sd": "SD-bl"}
        )
        fpath_amplitudes = os.path.join(
            output_folder, f"sz_sd_amplitudes_{get_datetime_for_fname()}.xlsx"
        )
        df_amplitudes.to_excel(fpath_amplitudes, index=False)
        print(f"Saved amplitudes file to {fpath_amplitudes}")
        # 3. Save baseline-trough difference amplitude
        df_bl_darkest = df_results[
            [
                "mouse_id",
                "event_uuid",
                "exp_type",
                "win_type",
                "y_bl",
                "y_trough",
                "dy_bl_trough",
                "extrapolated",
            ]
        ].sort_values(by=["exp_type", "win_type", "mouse_id"])
        df_bl_darkest = df_bl_darkest.rename(
            columns={
                "y_bl": "baseline",
                "y_trough": "darkest_postictal",
                "dy_bl_trough": "bl-darkest",
            }
        )
        fpath_bl_darkest = os.path.join(
            output_folder, f"bl-to-darkest-point_{get_datetime_for_fname()}.xlsx"
        )
        df_bl_darkest.to_excel(fpath_bl_darkest, index=False)
        print(f"Saved bl-through difference file to {fpath_bl_darkest}")
        # 4. Save peak-trough FWHM time
        df_peak_trough_fwhm = df_results[
            [
                "event_uuid",
                "mouse_id",
                "exp_type",
                "win_type",
                "dt_peak_FWHM",
                "extrapolated",
            ]
        ].sort_values(by=["exp_type", "win_type", "mouse_id"])
        fpath_peak_trough = os.path.join(
            output_folder, f"peak_fwhm_{get_datetime_for_fname()}.xlsx"
        )
        df_peak_trough_fwhm.to_excel(fpath_peak_trough, index=False)
        print(f"Saved peak-through FWHM file to {fpath_peak_trough}")
    # TODO: add specific analysis steps? (where df_results gets reshaped)
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fpath_stim_dset",
        type=str,
        default=None,
        help="Path to the stim dataset hdf5 file",
    )
    parser.add_argument(
        "--fpath_tmev_dset",
        type=str,
        default=None,
        help="Path to the tmev dataset hdf5 file",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save analysis results to Excel file",
    )
    parser.add_argument("--save_figs", action="store_true", help="Save figures")
    parser.add_argument(
        "--file_format", type=str, default="pdf", help="File format for figures"
    )
    args = parser.parse_args()
    fpath_tmev_assembled_traces = (
        args.fpath_tmev_dset
        if args.fpath_tmev_dset
        else open_file("Choose TMEV dataset")
    )
    fpath_stim_assembled_traces = (
        args.fpath_stim_dset
        if args.fpath_stim_dset
        else open_file("Choose stim dataset")
    )

    main(
        fpath_stim_dset=fpath_stim_assembled_traces,
        fpath_tmev_dset=fpath_tmev_assembled_traces,
        save_results=args.save_results,
        params=RecoveryAnalysisParams(),
    )
