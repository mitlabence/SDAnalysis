import os
import warnings
from datetime import datetime
import numpy as np
from locomotion_functions import extract_data_from_abf, find_rollovers, rotary_to_cm, create_total_distance, moving_average, get_episodes 
import loco_constants as loco_constants

class Experiment:
    def __init__(self, fpath, speed_cutoff: float = 0.2, episode_merge_threshold_s: float = 0.5):
        """Load and perform the analysis steps given an abf file with 3 channels: lfp, rotary encoder and LED output power signal.
        1. Load the data
        2. Extract the locomotion distance signal from the rotary encoder (convert to cm)
        3. Create total distance (i.e. eliminate rollover of rotary encoder)
        4. Calculate total absolute distance
            1. Smoothen the signal with sliding window averaging, 
            2. Calculate velocity (cm/s)
            3. High-pass filter it (take time points where velocity > threshold).
            4. Calculate accumulated absolute difference between these time points
        Args:
            fpath_abf (str): The path to the abf file
            speed_cutoff (float): The minimum speed a frame has to reach to be classified as "running" initially (cm/s)
            episode_merge_threshold_s (float): The threshold for merging episodes (in seconds). If two episodes are closer than this threshold, they are merged.
        """
        assert os.path.exists(fpath) and fpath.endswith(".abf")
        self.fpath_abf = fpath
        self.fname = os.path.basename(fpath)
        self.speed_cutoff = speed_cutoff
        # 1.
        self.t, self.lfp_y, self.rotary_y, self.stim_y, self.sample_rate = extract_data_from_abf(self.fpath_abf)
        # fix arduino signal for some recordings
        self.fix_arduino()
        # find rotary encoder rollovers
        self.rollovers = find_rollovers(self.rotary_y)
        # 2.
        self.rotary_y_cm = rotary_to_cm(self.rotary_y, self.rollovers, loco_constants.WHEEL_RADIUS)
        # 3.
        self.total_distance = create_total_distance(self.rotary_y_cm)
        # 4.
        # a.
        total_distance_smoothed = moving_average(self.total_distance, window_size=1000)  # 1s window
        # create speed
        speed = np.diff(total_distance_smoothed) / np.diff(self.t)
        speed = np.concatenate((speed, [speed[-1]]))  # pad with last value to match shape of total_distance
        self.speed = speed
        #speed_raw = np.diff(total_distance_smoothed) / np.diff(self.t)  # cm/s
        #speed_raw =  np.concatenate((speed_raw, [speed_raw[-1]]))  # pad with last value to match shape of total_distance
        #speed = speed_raw.copy()
        #speed[np.abs(speed) < self.speed_cutoff] = 0  # filter anything < 0.5 cm/s
        #self.total_distance_absolute = np.cumsum(np.abs(np.diff(total_distance_smoothed[speed != 0])))
        self.total_distance_absolute = np.concatenate([[0],np.cumsum(np.abs(np.diff(total_distance_smoothed)))])
        # filter out episodes < 1s and those not reaching a minimum speed 0.2 cm/s
        self.running = np.zeros(self.total_distance_absolute.shape, dtype=int)
        self.running[self.speed >= self.speed_cutoff] = 1
        idxs_stim_happens = np.where(self.stim_y > 0.05)[0]
        # convert led status voltage to %
        self.stim_y = self.stim_y / loco_constants.LED_MAX_VOLTAGE * 100.0  # convert to %
        if len(idxs_stim_happens) == 0:
            warnings.warn("No stim begin/end detected! Arduino was not working! Setting idx_stim_begin and idx_stim_end to None, and not creating pre/post locomotion quantifiers.")
            self.idx_stim_begin = None
            self.idx_stim_end = None
            self.idx_pre_begin = None
            self.idx_pre_end = None
            self.idx_post_begin = None
            self.idx_post_end = None
        else:
            self.idx_stim_begin = idxs_stim_happens[0]
            self.idx_stim_end = idxs_stim_happens[-1]
            # determine baseline and post begin and end frames, 5 minutes before and after stim
            self.idx_pre_begin = self.idx_stim_begin - 1 - 300 * self.sample_rate
            self.idx_pre_end = self.idx_stim_begin - 1
            self.idx_post_begin = self.idx_stim_end + 1
            self.idx_post_end = self.idx_stim_end + 1 + 300 * self.sample_rate
            # create pre/post locomotion quantifiers
            # absolute distance
            self.total_distance_absolute_pre = self.total_distance_absolute[self.idx_pre_end] - self.total_distance_absolute[self.idx_pre_begin]
            self.total_distance_absolute_post = self.total_distance_absolute[self.idx_post_end] - self.total_distance_absolute[self.idx_post_begin]
            # max speed
            self.max_speed_pre = np.max(self.speed[self.idx_pre_begin:self.idx_pre_end+1])
            self.max_speed_post = np.max(self.speed[self.idx_post_begin:self.idx_post_end+1])
            # number of episodes
            # first, merge episodes
            episode_merge_threshold = int(episode_merge_threshold_s * self.sample_rate)
            running_pre = self.running[self.idx_pre_begin:self.idx_pre_end+1]
            running_post = self.running[self.idx_post_begin:self.idx_post_end+1]
            list_episodes_pre = get_episodes(running_pre, True, episode_merge_threshold, return_begin_end_frames=True)
            list_epsisodes_post = get_episodes(running_post, True, episode_merge_threshold, return_begin_end_frames=True)
            self.n_episodes_pre = len(list_episodes_pre)
            self.n_episodes_post = len(list_epsisodes_post)
            running_filtered_pre = np.zeros(running_pre.shape, dtype=int)
            running_filtered_post = np.zeros(running_post.shape, dtype=int)
            for episode in list_episodes_pre:
                running_filtered_pre[episode[0]:episode[1]+1] = 1
            for episode in list_epsisodes_post:
                running_filtered_post[episode[0]:episode[1]+1] = 1
            self.running_percent_pre = 100.0*np.sum(running_filtered_pre) / len(running_filtered_pre)
            self.running_percent_post = 100.0*np.sum(running_filtered_post) / len(running_filtered_post)
    def fix_arduino(self):
        if self.fname == "2025_03_11_0000.abf":  # this SD stim recording had no Arduino signal. Fill the stim values manually
            self.stim_y[(self.t >= 332.053) & (self.t <= 352.053) ] = 5.0  # 20s stim, the beginning quite precisely known from LFP spike and comparing to other SD stims for this mouse
        elif self.fname == "2025_03_21_0006.abf":  # this sz mimic ctl recording started with high arduino signal, then jumps to pulse height for first pulse, then low etc.
            self.stim_y[:np.where(np.diff(self.stim_y) < -1)[0][0] + 1] = 0.0  # set to zero the first part of the recording, where the Arduino signal is not working


class ExperimentMetaData:
    def __init__(self, uuid: str, mouse_id: str, mouse_type: str, fname: str, date: datetime, exp_type: str, comment: str):
        self.uuid = uuid
        self.mouse_id = mouse_id
        self.mouse_type = mouse_type
        self.fname = fname
        self.date = date
        self.exp_type = exp_type
        self.comment = comment


class StimPattern:
    def __init__(self, pattern: str):
        """_summary_

        Args:
            pattern (str): a string in the form ledIndex1.peakDuration1.breakDuration1.numberOfPulses1.power1_ledIndex2.peakDuration2.breakDuration2.numberOfPulses2.power2_...
            example: '0.0.330000.1.0_2.1.30.75.400_2.4.332.8.680_2.4.332.30.730_2.4.499.20.780'
            meaning: break for 330000 ms (0 peak length), 75 1ms pulses with 30ms break at 40.0% power, LED index 2 (3rd led of machine, 0-5), etc.
        """
        self.pattern_raw = pattern
        self.pattern = self.decode_pattern(pattern)
    def __repr__(self):
        repr_str = f"StimPattern(pattern=("
        for step in self.pattern:
            repr_str += f"\n\tLED {step['ledIndex']} peakDuration: {step['peakDuration']} ms, breakDuration: {step['breakDuration']} ms, #pulses: {step['numberOfPulses']}, {step['power']}% power"
        return repr_str + "\n\t)"
    def decode_pattern(self, pattern: str):
        """_summary_

        Args:
            pattern (str): _description_

        Returns:
            List<dict>: A list of steps in the pattern, each step is a dictionary with keys:
                'ledIndex', 'peakDuration' (ms), 'breakDuration' (ms), 'numberOfPulses', 'power' (in %)
        """
        pattern = pattern.split('_')
        stim = []
        for step in pattern:
            step = step.split('.')
            ledIndex = int(step[0])
            peakDuration = float(step[1])
            breakDuration = float(step[2])
            numberOfPulses = int(step[3])
            power = float(step[4]) / 10.0
            stim.append({
                'ledIndex': ledIndex,
                'peakDuration': peakDuration,
                'breakDuration': breakDuration,
                'numberOfPulses': numberOfPulses,
                'power': power
            })
        return stim
    def total_number_of_pulses(self):
        """_summary_: returns the total number of pulses in the pattern
        """
        return sum([step['numberOfPulses'] for step in self.pattern if step["peakDuration"] > 0 and step["power"] > 0])
        