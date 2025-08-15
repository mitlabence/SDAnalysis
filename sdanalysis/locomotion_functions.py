"""
locomotion_functions.py - Functions for locomotion analysis
"""
from typing import Optional
import warnings
import numpy as np


def apply_threshold(speed_trace, episodes, temporal_threshold, amplitude_threshold):
    """
    Given a speed trace and a list of tuples (i_begin_frame, i_end_frame) marking running episodes, this function discards those that
    a.) are shorter than the defined temporal threshold (in units of frames),
    OR
    b.) the amplitude of the absolute trace does not reach the amplitude threshold during the episode.
    Returns the filtered episodes.
    """

    discard_list = []
    # tuple of (i_begin, i_end). Assume [i_begin:i_end+1] is correct, see get_episodes()
    for i_episode, episode in enumerate(episodes):
        episode_trace = speed_trace[episode[0] : episode[1] + 1]
        # filter by temporal threshold
        if len(episode_trace) < temporal_threshold:
            # print(f"{len(episode_trace)}")
            if i_episode not in discard_list:
                discard_list.append(i_episode)
        # filter by amplitude threshold
        if max(np.abs(episode_trace)) < amplitude_threshold:
            if i_episode not in discard_list:
                discard_list.append(i_episode)
    discard_list = sorted(discard_list)

    # discard components
    episodes_filtered = [
        episodes[i] for i in range(len(episodes)) if i not in discard_list
    ]
    return episodes_filtered


def get_episodes(
    segment,
    merge_episodes=False,
    merge_threshold_frames=8,
    return_begin_end_frames=False,
):
    """Given a binary trace (0 is rest, 1 is locomotion), return the (beginning, end) frames of each locomotion episode.
    If merge_episodes=True, also

    Parameters
    ----------
    segment : _type_
        _description_
    merge_episodes : bool, optional
        _description_, by default False
    merge_threshold_frames : int, optional
        _description_, by default 8
    return_begin_end_frames : bool, optional
        Whether to return the number of episodes, or the episode beginning and end frames.
        If set to return indices (True), then (i_begin, i_end) are both inclusive in 0-indexing! By default False

    Returns
    -------
    _type_
        _description_
    """
    #

    n_eps = 0
    episode_lengths = []  # in frame units
    episodes = []
    n_episodes = 0
    current_episode_len = 0

    episode_begin = 0
    episode_end = 0

    # algorithm: detect episode begin and episode end. record it in list

    # check current and next element for end of a episode: ...100...
    for i_frame in range(len(segment) - 1):
        if segment[i_frame] == 1:  # current frame is part of an episode
            # increase current episode length
            # check if beginning of an episode or segment starts with an episode
            if i_frame == 0 or segment[i_frame - 1] == 0:
                episode_begin = i_frame
            current_episode_len += 1
            if segment[i_frame + 1] == 0:  # episode ends with next frame
                n_episodes += 1
                episode_lengths.append(current_episode_len)
                episodes.append((episode_begin, i_frame))
                current_episode_len = 0
    if segment[-1] == 1:  # check if there is one episode that does not end
        n_episodes += 1
        # add last segment to segments list
        current_episode_len += 1
        episode_lengths.append(current_episode_len)
        episodes.append((episode_begin, len(segment) - 1))
        current_episode_len = 0

    assert current_episode_len == 0
    if merge_episodes:
        if len(episodes) < 2:  # single (or zero) episode cannot be merged
            if return_begin_end_frames:
                return episodes
            else:
                return [ep[1] - ep[0] + 1 for ep in episodes]

        # merge episodes that are close to each other
        episodes_merged = []

        episode_begin = episodes[0][0]
        episode_end = episodes[0][1]
        # starting with second episode, check if current episode can be merged with previous. If yes, update episode_end.
        # If not, add previous episode to list, update episode_begin and episode_end to current episode values

        for i_episode in range(1, len(episodes)):
            current_episode_begin = episodes[i_episode][0]
            current_episode_end = episodes[i_episode][1]

            delta = current_episode_begin - episode_end

            if delta <= merge_threshold_frames:  # merge current episode to previous one
                episode_end = current_episode_end
            else:  # add previous episode to list, start with current episode
                episodes_merged.append((episode_begin, episode_end))
                episode_begin = current_episode_begin
                episode_end = current_episode_end
        # add last segment to list
        episodes_merged.append((episode_begin, episode_end))
        if return_begin_end_frames:
            return episodes_merged
        else:
            episode_lengths_merged = [ep[1] - ep[0] + 1 for ep in episodes_merged]
            return episode_lengths_merged
    if return_begin_end_frames:
        return episodes
    else:
        return episode_lengths  # len() shows n_episodes


def calculate_avg_speed(speed_trace, mask: Optional[np.array] = None):
    """Given a speed trace list or numpy array, calculate the average absolute speed.
    Frames where mask != 1 are ignored.

    Parameters
    ----------
    speed_trace : iterable, list[float] or np.array(float)
        A 1D list or numpy array of speed values (float)
    mask : Optional, np.array(int)
        A 1D list with same shape as speed_trace, where 1 indicates a frame to be included in the
        calculation, and 0 indicates a frame to be ignored. If None, all frames are included.
        If values other than 0 and 1 are present, they are interpreted as a 0.
        By default None.
    Returns
    -------
    float
        The average apsolute speed over the whole trace.
    Raises
    ------
    ValueError
        If speed_trace and mask do not have the same shape.
    """
    speed_trace = np.array(speed_trace)
    if mask is None:
        mask = np.full(shape=speed_trace.shape, fill_value=True)
    else:
        mask = np.array(mask == 1)
        
    if mask is not None and speed_trace.shape != mask.shape:
        raise ValueError(
            "calculate_avg_speed(): speed_trace and mask must have the same shape!"
        )
    return np.mean(np.abs(speed_trace[mask]))


def calculate_max_speed(speed_trace):
    """Given a speed trace list or numpy array, calculate the absolute maximum speed.

    Parameters
    ----------
    speed_trace : iterable, list[float] or np.array(float)
        A 1D list or numpy array of speed values (float)
    Returns
    -------
    float
        The maximum absolute speed. This can be the absolute value of a negative speed reached as well!
    """
    speed_trace = np.array(speed_trace)
    # np.median(np.sort(speed_trace)[floor(0.95*len(speed_trace)):])
    return np.max(np.abs(speed_trace))


def get_trace_delta(trace, i_begin, i_end_exclusive):
    """Given a monotonously changing trace, get the change during the segment starting at frame i_begin, and ending one frame before i_end_exclusive.
    I.e. returns trace[i_begin:i_end_exclusive] [-1] - [0].
    Parameters
    ----------
    trace : 1D iterable
        The monotonously changing complete trace
    i_begin : int
        The first (0-indexed) frame to include.
    i_end_exclusive : int
        One after the last (0-indexed) frame to include. Reflects a[x:y] indexing conventions.
    Returns
    -------
    float
        The change during the segment.
    """
    trace = np.array(trace)
    if not (np.all(trace[1:] >= trace[:-1]) or np.all(trace[1:] <= trace[:-1])):
        warnings.warn("get_trace_delta(): trace is not monotonous!")
    trace_cut = trace[i_begin:i_end_exclusive]
    return trace_cut[-1] - trace_cut[0]



# functions related to the SLE experiments performed at a different setup.

import pyabf
import math

# rotary encoder specific parameters, measured using reference dataset
VMAX = 4.102783203125
VMIN = 0.0103759765625


def extract_data_from_abf(fpath_abf: str = None):
    """
    Given an abf file with channels LFP, rotary encoder and LED status signal, extract the data from the file. The time stamps are the same for all channels (t variable)

    Args:
        fpath_abf (str, optional): _description_. Defaults to None.

    Returns:
        t: The time stamps of the data in s (starting from 0)
        lfp_y: The LFP data in uV(?)
        rotary_y: The raw rotary encoder data in V (ranges from approx. 0-4.1 V)
        stim_y: The LED status signal (analog signal, noisy, the amplitude correlates with the LED set power, theoretical range 0-5V)
    """
    abf = pyabf.ABF(fpath_abf)
    framerate = abf.sampleRate  # in Hz
    abf.setSweep(0, channel=0)
    t = abf.sweepX  # t is same for all channels
    lfp_y = abf.sweepY
    abf.setSweep(0, channel=1)
    rotary_y = abf.sweepY
    abf.setSweep(0, channel=2)
    stim_y = abf.sweepY
    return t, lfp_y, rotary_y, stim_y, framerate

def find_rollovers(rotary_y: np.ndarray):
    """
    Given a raw rotary encoder signal, find the points where the signal rolls over from 0 to 5V (marked with 1) or vice versa (marked with -1).

    Args:
        rotary_y (np.ndarray): _description_
    Returns:
        np.ndarray: an array the same length as rotary_y, with 1 where the signal rolls over from 0 to 5V (i.e. the first low value after reaching 5 V), 
        -1 where it rolls over from 0V to 5V (i.e. the first high value after reaching 0 V)
    """
    idx_forward_skips = np.where(np.diff(rotary_y) < -0.5)[0] + 1  # diff shows a[i+1] - a[i], i.e. last point before rollover
    idx_backward_skips = np.where(np.diff(rotary_y) > 0.5)[0] + 1
    rollovers = np.zeros(len(rotary_y))
    rollovers[idx_forward_skips] = 1
    rollovers[idx_backward_skips] = -1
    return rollovers


def rotary_to_cm(rotary_y: np.ndarray, rollovers: np.ndarray, wheel_radius: float = 5):
    """
    Given the raw data of the rotary encoder signal, convert it to cm using the setup parameters (wheel radius)

    Args:
        rotary_y (np.ndarray): The raw rotary encoder signal
        rollovers (np.ndarray): Same shape as rotary_y, 1 where where the signal rolls over from 5V to 0V (i.e. forward movement), -1 where it rolls over from 0V to 5V
    Returns:
        _type_: _description_
    """
    # if there was no rollover, use the estimate VMIN and VMAX
    n_rollovers = np.sum(np.abs(rollovers) == 1)
    voltage_min = VMIN
    voltage_max = VMAX
    if n_rollovers > 0:
        # If there was no rollover forward or backward, substitute it with estimated VMIN or VMAX; otherwise, 
        # take the mean of all minima and maxima. These are found as:
        #    maxima: either index where rollovers == -1, or index - 1 where rollovers == 1
        #    minima: either index where rollovers == 1, or index - 1 where rollovers == -1
        n_rollovers_forward = np.sum(rollovers == 1)
        n_rollovers_backward = np.sum(rollovers == -1)
        # already handled the case where both are 0
        if n_rollovers_forward == 0:  # rollovers backward mean 0 V to 5V. rollovers is -1 where it rolls over from 0V to 5V and points at the high value.
            # so to get minimum voltages, need to check element before the rollover
            voltage_max = np.mean(rotary_y[rollovers == -1])
            voltage_min = np.mean(rotary_y[np.where(rollovers == -1)[0] - 1])
        elif n_rollovers_backward == 0:  # rollovers forward mean 5 V to 0 V. rollovers is 1 where it rolls over from 5V to 0V and points at the low value.
            # so to get maximum voltages, need to check element before the rollover
            voltage_min = np.mean(rotary_y[rollovers == 1])
            voltage_max = np.mean(rotary_y[np.where(rollovers == 1)[0] - 1])
        else: # both forward and backward rollovers
            voltage_max = np.mean(np.concatenate((rotary_y[rollovers == -1], rotary_y[np.where(rollovers == 1)[0] - 1])))
            voltage_min = np.mean(np.concatenate((rotary_y[rollovers == 1], rotary_y[np.where(rollovers == -1)[0] - 1])))
        # for the rest of the rounds, use the rollover points as minimum and maximum
    rotary_y_cm = 2*wheel_radius*math.pi*(rotary_y - voltage_min) / (voltage_max - voltage_min)
    return rotary_y_cm


def create_total_distance(rotary_y_cm: np.ndarray):
    """
    Given the rotary encoder data in cm (see rotary_to_cm()), create the total distance covered by the animal. This is done by adding the distance covered in each time step to the distance covered in the previous time step.

    Args:
        rotary_y_cm (np.ndarray): _description_
    Returns:
        _type: _description_
    """
    total_distance = rotary_y_cm.copy()  # avoid overwriting numpy array (I remember passing by reference in case of numpy arrays?)
    total_distance -= total_distance[0]  # start from 0 cm
    #idx_forward_skips = np.where(np.diff(rotary_y_cm) < -0.5)[0]
    #idx_backward_skips = np.where(np.diff(rotary_y_cm) > 0.5)[0]
    idx_skips = np.where(np.abs(np.diff(rotary_y_cm)) > 0.5)[0] + 1  # add 1 to avoid off-by-one error
    # for each break point, shift the whole "future" dataset by the distance of the break
    # estimate step size to be the same as the step before the break
    for idx in idx_skips:
        # shift the whole future to start at total_distance[idx-1] + step_size
        total_distance[idx:] += total_distance[idx-1] - total_distance[idx]
    return total_distance


def moving_average(total_distance: np.ndarray, window_size: int = 1000):
    """
    Given the total distance data (see create_total_distance), create a moving average of the data. The window size is defined by the SAMPLING_FREQUENCY constant.
    Args:
        total_distance (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    # create moving average
    window = np.ones(window_size)/window_size
    total_distance_smoothed = np.convolve(total_distance, window, mode='valid')  # cm
    # create moving average window, pad with zeros at the beginning, pad with last value at the end (to match shape of total_distance)
    total_distance_smoothed = np.concatenate((np.full(len(window)//2, total_distance_smoothed[0]), total_distance_smoothed, np.full(len(window)//2-1, total_distance_smoothed[-1])))
    total_distance_smoothed = total_distance_smoothed - total_distance_smoothed[0]  # start from 0 cm
    assert len(total_distance_smoothed) == len(total_distance)
    return total_distance_smoothed


def downsample(total_distance: np.ndarray, t: np.ndarray, sampling_frequency: int = 1000, downsampled_frequency: int = 100):
    """
    Downsample the total distance data to 100 Hz. This is done by linear interpolation.

    Args:
        total_distance (np.ndarray): _description_
        t (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    assert sampling_frequency >= downsampled_frequency
    t_downsampled = np.linspace(0, t[-1], math.floor(len(t)/sampling_frequency*downsampled_frequency))
    total_distance_downsampled = np.interp(t_downsampled, t, total_distance)
    return t_downsampled, total_distance_downsampled

# TODO: this function (get_episodes) is copied from SDAnalysis. Move it to a common location (in a third package?)
def get_episodes(
    segment,
    merge_episodes=False,
    merge_threshold_frames=8,
    return_begin_end_frames=False,
):
    """Given a binary trace (0 is rest, 1 is locomotion), return the (beginning, end) frames of each locomotion episode.
    If merge_episodes=True, also

    Parameters
    ----------
    segment : _type_
        _description_
    merge_episodes : bool, optional
        _description_, by default False
    merge_threshold_frames : int, optional
        _description_, by default 8
    return_begin_end_frames : bool, optional
        Whether to return the number of episodes, or the episode beginning and end frames.
        If set to return indices (True), then (i_begin, i_end) are both inclusive in 0-indexing! By default False

    Returns
    -------
    _type_
        _description_
    """
    #

    n_eps = 0
    episode_lengths = []  # in frame units
    episodes = []
    n_episodes = 0
    current_episode_len = 0

    episode_begin = 0
    episode_end = 0

    # algorithm: detect episode begin and episode end. record it in list

    # check current and next element for end of a episode: ...100...
    for i_frame in range(len(segment) - 1):
        if segment[i_frame] == 1:  # current frame is part of an episode
            # increase current episode length
            # check if beginning of an episode or segment starts with an episode
            if i_frame == 0 or segment[i_frame - 1] == 0:
                episode_begin = i_frame
            current_episode_len += 1
            if segment[i_frame + 1] == 0:  # episode ends with next frame
                n_episodes += 1
                episode_lengths.append(current_episode_len)
                episodes.append((episode_begin, i_frame))
                current_episode_len = 0
    if segment[-1] == 1:  # check if there is one episode that does not end
        n_episodes += 1
        # add last segment to segments list
        current_episode_len += 1
        episode_lengths.append(current_episode_len)
        episodes.append((episode_begin, len(segment) - 1))
        current_episode_len = 0

    assert current_episode_len == 0
    if merge_episodes:
        if len(episodes) < 2:  # single (or zero) episode cannot be merged
            if return_begin_end_frames:
                return episodes
            else:
                return [ep[1] - ep[0] + 1 for ep in episodes]

        # merge episodes that are close to each other
        episodes_merged = []

        episode_begin = episodes[0][0]
        episode_end = episodes[0][1]
        # starting with second episode, check if current episode can be merged with previous. If yes, update episode_end.
        # If not, add previous episode to list, update episode_begin and episode_end to current episode values

        for i_episode in range(1, len(episodes)):
            current_episode_begin = episodes[i_episode][0]
            current_episode_end = episodes[i_episode][1]

            delta = current_episode_begin - episode_end

            if delta <= merge_threshold_frames:  # merge current episode to previous one
                episode_end = current_episode_end
            else:  # add previous episode to list, start with current episode
                episodes_merged.append((episode_begin, episode_end))
                episode_begin = current_episode_begin
                episode_end = current_episode_end
        # add last segment to list
        episodes_merged.append((episode_begin, episode_end))
        if return_begin_end_frames:
            return episodes_merged
        else:
            episode_lengths_merged = [ep[1] - ep[0] + 1 for ep in episodes_merged]
            return episode_lengths_merged
    if return_begin_end_frames:
        return episodes
    else:
        return episode_lengths  # len() shows n_episodes
