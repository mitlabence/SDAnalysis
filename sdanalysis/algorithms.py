"""
algorithms.py  - misc. algorithms
"""

import numpy as np
import pandas as pd


def get_downsampling_indices(
    time_stamps: np.array, downsampled_time_stamps: np.array
) -> np.array:
    """
    Given an array of time_stamps (length N) and an array of downsampled_time_stamps (length n < N),
    return the indices of data/time_stamps that mark the closest entries to the downsampled
    time stamps.
    Example:
    data_downsampled = data[get_downsampling_indices(time_stamps, downsampled_time_stamps)]
    Args:
        data (np.array): _description_
        time_stamps (np.array): _description_
        downsampled_time_stamps (np.array): _description_

    Returns:
        np.array: The indices of time_stamps that are closest to each entry in
        downsampled_time_stamps.
    """
    idx_nearest = np.array(
        [
            np.abs(time_stamps - t_downsampled).astype(np.float32).argmin()
            for t_downsampled in downsampled_time_stamps
        ],
        dtype=int,
    )  # change to float32 to avoid precision errors
    return idx_nearest


def matlab_smooth(array):
    """
    This function tries to imitate the matlab smooth function:
    yy(0) = y(0)
    yy(1) = (y(0) + y(1) + y(2))/3
    yy(2) = (y(0) + y(1) + y(2) + y(3) + y(4))/5
    yy(3) = (y(1) + y(2) + y(3) + y(4) + y(5))/5
    ...
    yy(end-2) = (y(end-4) + y(end-3) + y(end-2) + y(end-1) + y(end))/5
    yy(end-1) = (y(end-2) + y(end-1) + y(end))/3
    yy(end) = y(end)

    Args:
        array (_type_): _description_
    """
    smoothed = np.convolve(array, np.ones(5) / 5, mode="same")
    # first and last elements are the same
    smoothed[0] = array.iloc[0]
    smoothed[-1] = array.iloc[-1]
    # second and second to last elements are the average of the first three and last three
    smoothed[1] = np.mean(array[:3])
    smoothed[-2] = np.mean(array[-3:])
    return smoothed


def speed_to_meters_per_second(speed_data, time_stamps_ms) -> pd.Series:
    """
    Convert the speed data (sampled by LabView) to meters per second.

    Args:
        speed_data (pd.DataFrame): _description_
        time_stamps_ms (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    speed_m_s = speed_data.copy()
    conversion_factor = 100.0
    speed_m_s = speed_m_s / conversion_factor
    # use time stamps to get the time between frames
    speed_m_s.iloc[0] = 0.0
    time_diff_ms = time_stamps_ms.diff()  # delta_t = t_i - t_{i-1}, first is NaN
    speed_m_s[1:] = speed_m_s[1:] / time_diff_ms[1:]
    return speed_m_s
