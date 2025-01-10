"""
linear_locomotion.py - Module to match LabView and ND2 time stamps to get linear motion data. 
"""

import warnings
import numpy as np
import pandas as pd
from nd2_time_stamps import ND2TimeStamps
from lv_data import LabViewData
from lv_time_stamps import LabViewTimeStamps
from algorithms import (
    matlab_smooth,
    get_downsampling_indices,
    speed_to_meters_per_second,
)


class LinearLocomotion:
    """
    Class to match LabView and ND2 time stamps to get linear motion data.
    All time stamps are in seconds by default; see @property functions for milliseconds.
    Attributes:
        lv_data (pd.DataFrame): The data from the LabView data file (data.txt) with the same
            time stamps as lv_time_stamps_belt
        lv_time_stamps_belt (pd.Series): The time stamps of the belt from the LabView
            timestamp data (time.txt)
        lv_time_stamps_scanner (pd.DataFrame): The time stamps of the scanner from the LabView
            timestamp data (time.txt).
        scanner_time_stamps (pd.Series): The time stamps of the scanner in seconds (from
            the _nik.txt file)


    """

    def __init__(
        self,
        nd2_time_stamps: ND2TimeStamps,
        lv_time_stamps: LabViewTimeStamps,
        lv_data: LabViewData,
        **kwargs,
    ):
        # helpful arguments for tests
        break_after_matching = kwargs.get("break_after_matching", False)
        break_after_arduino_corr = kwargs.get("break_after_arduino_corr", False)
        break_after_belt_corr = kwargs.get("break_after_belt_corr", False)
        # time stamps are in seconds by default; see @property functions for milliseconds
        self.scanner_time_stamps = (
            nd2_time_stamps.time_stamps
        )  # TODO: this will later become Series, so do not assign here? (misleading)
        self.time_column = (
            nd2_time_stamps.time_column
        )  # FIXME: should remove this once pipeline
        # is ready, and use SW Time [s] as intended
        self.lv_time_stamps_scanner = lv_time_stamps.scanner_time_stamps

        self.n_missed_frames = self._get_n_missed_frames(
            self.scanner_time_stamps, self.lv_time_stamps_scanner
        )
        self.lv_time_stamps_belt, self.n_missed_cycles = (
            self._get_raw_belt_time_stamps_missed_cycles(
                lv_time_stamps_belt=lv_time_stamps.time_stamps["belt_time_stamps"],
                lv_data_time_total_ms=lv_data.data["time_total_s"],
            )
        )
        self.scanner_time_stamps, self.source_scanner_time_stamps = (
            self._create_scanner_time_stamps(
                self.scanner_time_stamps[self.time_column],
                self.lv_time_stamps_scanner,
            )
        )
        # TODO: check for NaNs probably not needed
        n_nans = np.isnan(self.scanner_time_stamps).sum()
        if n_nans > 0:
            raise ValueError(f"NaN found ({n_nans} total) among scanner time stamps")
        if not self.scanner_time_stamps.is_unique:
            # TODO: implement interpolation from beltMatchToNikonStampsExpProps.m line 204
            raise ValueError("Non-unique scanner time stamps")
        if not self.scanner_time_stamps.is_monotonic_increasing:
            raise ValueError("Non-monotonic increasing scanner time stamps")
        self.i_belt_start, self.i_belt_stop = self._get_first_last_frames_belt(
            self.lv_time_stamps_belt, self.lv_time_stamps_scanner
        )
        # cut labview data to match the scanner recording begin and end
        self.lv_data = lv_data.data.iloc[self.i_belt_start : self.i_belt_stop + 1]
        # cut labview belt time stamps to match the scanner recording begin and end
        self.lv_time_stamps_belt = self.lv_time_stamps_belt.iloc[
            self.i_belt_start : self.i_belt_stop + 1
        ]
        # reset indices
        self.lv_data = self.lv_data.reset_index(drop=True)
        self.lv_time_stamps_belt = self.lv_time_stamps_belt.reset_index(drop=True)
        self.lv_time_stamps_scanner = self.lv_time_stamps_scanner.reset_index(drop=True)
        # start time stamps with 0
        self.lv_time_stamps_belt = (
            self.lv_time_stamps_belt - self.lv_time_stamps_belt.iloc[0]
        )
        # replace time stamps in belt
        self.lv_data["time_total_s"] = self.lv_time_stamps_belt
        if break_after_matching:
            return
        # start various parameters (time, distance, rounds ...) at 0
        self.lv_data = self._set_initial_values(self.lv_data)
        # correct arduino artifacts
        self.lv_data = self._correct_arduino_artifacts(self.lv_data)
        if break_after_arduino_corr:
            return
        # correct stripes per round
        # FIXME: add this back in, remove "if zone > 2 break" in _correct_belt_length()
        # self.lv_data = self._correct_stripes_per_round(self.lv_data, n_zones_expected=3)
        # correct belt length
        self.lv_data = self._correct_belt_length(self.lv_data)
        if break_after_belt_corr:
            return
        # add binary "running" mask
        self.lv_data["running"] = self._get_running_mask(self.lv_data)
        # convert to m/s
        self.lv_data["speed"] = speed_to_meters_per_second(
            self.lv_data["speed"], self.lv_data["time_total_s"]
        )
        self.idx_downsample = get_downsampling_indices(
            self.lv_data["time_total_s"], self.scanner_time_stamps
        )
        self.lv_data_downsampled = pd.DataFrame(
            {
                "time_total_s": self.scanner_time_stamps,
                "round": self.lv_data["round"][self.idx_downsample].reset_index(drop=True),
                "speed": self.lv_data["speed"][self.idx_downsample].reset_index(drop=True),
                "distance_per_round": self.lv_data["distance_per_round"][
                    self.idx_downsample
                ].reset_index(drop=True),
                "total_distance": self.lv_data["total_distance"][self.idx_downsample].reset_index(drop=True),
                "running": self.lv_data["running"][self.idx_downsample].reset_index(drop=True),
            }
        )

    @property
    def duration(self):
        """
        Get the duration of the recording in minutes.

        Returns:
            float: The duration of the recording in minutes.
        """
        return self.scanner_time_stamps.iloc[-1] / 60.0  # time stamps are in seconds

    @property
    def n_frames(self):
        """
        Get the number of frames recorded.

        Returns:
            int: The number of frames recorded.
        """
        return len(self.scanner_time_stamps)

    @property
    def imaging_frequency(self):
        """
        Get the average imaging frequency in Hz.

        Returns:
            float: The average imaging frequency in Hz.
        """
        return self.n_frames / (self.duration * 60.0)  # n_frames / duration in seconds

    @property
    def scanner_time_stamps_ms(self):
        """
        Get the scanner time stamps in milliseconds. In the matlab code, this is equivalent to
        tsscn.

        Returns:
            pd.Series: The scanner time stamps (starting with 0) in milliseconds. It is equivalent
            to tsscn in the matlab code.
        """
        return self._convert_to_ms(self.scanner_time_stamps)

    def _convert_to_ms(self, df: pd.DataFrame):
        """
        Convert all time-related columns to milliseconds.

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = df.copy()
        if isinstance(df, pd.Series):
            return df * 1000.0
        time_columns = [col for col in df.columns if "time" in col]
        for col in time_columns:
            df[col] = df[col] * 1000.0
        # rename affected columns to reflect unit change
        df.columns = [
            col.replace("_s", "_ms") if "time" in col else col for col in df.columns
        ]
        return df

    def _convert_to_s(self, df: pd.DataFrame):
        """
        Find time-related columns, and convert them to seconds.

        Args:
            df (pd.DataFrame): _description_
        """
        if isinstance(df, pd.Series):
            return df / 1000.0
        df = df.copy()
        time_columns = [col for col in df.columns if "time" in col]
        for col in time_columns:
            df[col] = df[col] / 1000.0
        return df

    def _get_n_missed_frames(
        self, time_stamps_nik: pd.DataFrame, time_stamps_lv: pd.DataFrame
    ) -> int:
        """
        Match time stamps from LabView and ND2 files.
        Args:
            time_stamps_nik (pd.DataFrame): Time stamps from the _nik.txt file (Nikon time stamps)
            time_stamps_lv (pd.DataFrame): Time stamps for the scanner (resonant/galvano) from the
            .txt file (LabView scanner time stamps)
        """
        # get the possible number of missed frames in the labview based on the actual number of
        # frames recorded (could happen, for example, if labview stops before nd2 recording stops)
        n_missed_frames = len(time_stamps_nik) - len(time_stamps_lv)
        if n_missed_frames < 0:
            raise ValueError("More frames in LabView than in ND2")
        return n_missed_frames

    def _get_raw_belt_time_stamps_missed_cycles(
        self,
        lv_time_stamps_belt: pd.Series,
        lv_data_time_total_ms: pd.Series,
    ):
        """


        Args:
            lv_time_stamps_belt (pd.Series): _description_
            lv_time_stamps_scanner (pd.Series): _description_
            scanner_time_stamps (pd.Series): _description_
            lv_data_time_total_ms (pd.Series): _description_

        Returns:
            tuple(pd.Series, int, int):

        """
        if (lv_time_stamps_belt > 0).any():  # check if first column is not zeros
            # non-zero entries in first column (corresponding to "belt" time stamps)
            lv_time_stamps_belt = lv_time_stamps_belt[lv_time_stamps_belt > 0]
            n_missed_cycles = self._get_n_missed_cycles(
                lv_data_time_total_ms, lv_time_stamps_belt
            )
            if (
                n_missed_cycles > 10
            ):  # TODO: beltMatchToNikonStampsExpProps.m line 118, use abs?
                warnings.warn(
                    f"More than 10 missed cycles in LabView data: {n_missed_cycles}"
                )
                offset = (
                    lv_time_stamps_belt[0]
                    if len(lv_time_stamps_belt) == 1
                    else lv_time_stamps_belt[1]
                )
                lv_time_stamps_belt = (
                    lv_data_time_total_ms - lv_data_time_total_ms.iloc[0] + offset
                )
            return lv_time_stamps_belt.reset_index(drop=True), n_missed_cycles
        warnings.warn("No belt time stamps found in LabView time stamps data")
        n_missed_cycles = np.nan
        return lv_data_time_total_ms.reset_index(drop=True), n_missed_cycles

    def _get_n_missed_cycles(
        self, lv_belt_data_time: pd.Series, lv_time_stamps: pd.Series
    ):
        """
        Get the number of missed cycles (belt frames) in the LabView data.
        It is the discrepancy between the .txt and the time.txt LV files.
        Args:
            lv_belt_data_time (pd.Series): The time stamps from the LabView data file (.txt)
            lv_time_stamps (pd.Series): The time stamps from the LabView timestamp file (time.txt)
        """
        return len(lv_belt_data_time) - len(lv_time_stamps)

    def _get_first_last_frames_belt(
        self, lv_time_stamps_belt: pd.Series, lv_time_stamps_scanner: pd.Series
    ):
        """
        Given the scanner time stamps and belt time stamps, find the first and last frames of the
        belt recording that were recorded during the scanner recording.

        Args:
            lv_time_stamps_belt (pd.Series): _description_
            lv_time_stamps_scanner (pd.Series): _description_

        Returns:
            i_begin: The first frame index such that
                lv_time_stamps_belt[i_begin] > lv_time_stamps_scanner[0]
            i_end: The last frame index such that
                lv_time_stamps_belt[i_end] < lv_time_stamps_scanner[-1]
        """
        # assume belt recording started first, then scanner recording.
        # the start index of the belt recording is the first frame after the first scanner frame
        i_begin = lv_time_stamps_belt[
            lv_time_stamps_belt >= lv_time_stamps_scanner.iloc[0]
        ].index[0]
        # the stop index of the belt recording is the first frame before the last scanner frame
        i_end = lv_time_stamps_belt[
            lv_time_stamps_belt <= lv_time_stamps_scanner.iloc[-1]
        ].index[-1]
        return i_begin, i_end

    def _create_scanner_time_stamps(
        self, time_stamps_nik: pd.Series, lv_time_stamps_scanner: pd.Series
    ):
        """
        Create time stamps starting with 0 for the scanner frames.
        Either the labview time stamps for the scanner frames
        are used (if none of the frames was missed), or the nikon (Software) time stamps.
        Args:
            time_stamps_nik (pd.Series): The time stamps from the nikon recording
            lv_time_stamps_scanner (pd.Series): The time stamps from the scanner recording
        Returns:
            pd.Series: The time stamps for the scanner frames
            str: The source of the time stamps: "labview" or "nikon"
        """
        n_time_stamps_nik = len(time_stamps_nik)
        n_time_stamps_lv = len(lv_time_stamps_scanner)
        if n_time_stamps_nik > n_time_stamps_lv:
            # not all frames were detected in the labview recording. -> use nikon time stamps
            # TODO: same as n_missed_frames?
            return time_stamps_nik - time_stamps_nik[0], "nikon"
        if n_time_stamps_nik < n_time_stamps_lv:
            raise ValueError("More frames registered in LabView than in ND2.")
        # equal number of frames: prefer labview time stamps
        return lv_time_stamps_scanner - lv_time_stamps_scanner[0], "labview"

    def _correct_arduino_artifacts(
        self,
        labview_data: pd.DataFrame,
        threshold_p: int = 700,
        threshold_n: int = -200,
    ):
        """
        Correct Arduino-related artifacts (artificial speed values) in the arduino data.
        The origin of these artifacts is lost over time; it might be a mouse-sensor related
        issue.
        Args:
            labview_data (pd.DataFrame): The LabView data.
            threshold_p (int): The positive threshold for the speed.
            threshold_n (int): The negative threshold for the speed.
        """
        labview_data = labview_data.copy()
        # find all values above crossing one threshold
        idx = (labview_data["speed"] > threshold_p) | (
            labview_data["speed"] < threshold_n
        )
        idx = labview_data["speed"][
            idx
        ].index  # convert to Int64Index array (it is sorted)
        # correct all values above threshold
        for i in idx:
            # correct the speed, distance, distance per round.
            if i == 0:  # first frame, cannot use a frame before
                # only correct the speed (rest should be 0)
                labview_data.loc[i, "speed"] = (
                    threshold_p if labview_data.loc[i, "speed"] > 0 else threshold_n
                )
            else:
                # correct speed: copy last time step
                labview_data.loc[i, "speed"] = labview_data.loc[i - 1, "speed"]
                # correct distance: remove delta stemming from outlier speed, and
                # add the delta from the last time step
                # idea: dist[i] = dist[i-1] + d[i] where
                # d[i] = speed[i] * dt (the change for timestep i)
                # If we replace speed[i] -> speed[i-1], then d[i] -> d[i-1]
                # so dist[i] = dist[i-1] + d[i-1] is the new formula
                #  = ( dist[i-1] + d[i] ) - d[i] + d[i-1]
                labview_data.loc[i:, "total_distance"] = (
                    labview_data.loc[i:, "total_distance"]
                    - labview_data.loc[i, "total_distance"]
                    + labview_data.loc[i - 1, "total_distance"]
                )
                # correct distance per round in same way but only for the current round
                i_round = labview_data.loc[i, "round"]
                i_round_last_idx = labview_data[labview_data["round"] == i_round].index[
                    -1
                ]
                labview_data.loc[i:i_round_last_idx, "distance_per_round"] = (
                    labview_data.loc[i : i_round_last_idx + 1, "distance_per_round"]
                    - labview_data.loc[i, "distance_per_round"]
                    + labview_data.loc[i - 1, "distance_per_round"]
                )
        return labview_data

    def _correct_stripes_per_round(
        self, labview_data: pd.DataFrame, n_zones_expected: int = 3
    ):
        """
        Correct various bugs regarding the stripes.
        1. When a new round is started, the time frame before it the stripe increases, and also
        the stripe per round. If there are 3 zones, for example, the last frame in a given round has
        stripe per round = 3, whereas it should be in [0, 1, 2].
        Args:
            labview_data (pd.DataFrame): _description_
        """
        # get the indices of frames where new rounds start
        idxs_new_round = np.where(
            np.diff(
                pd.concat(
                    [
                        pd.Series([labview_data["round"].iloc[0]]),
                        labview_data["round"],
                    ]
                )
            )
        )[0]
        if labview_data["stripes_per_round"].max() > n_zones_expected - 1:
            # correct the stripes per round for frames before new round
            for i_round in idxs_new_round:
                # make sure that we catch the true new round (new stripe as well)
                assert (
                    labview_data.loc[i_round, "stripes_total"]
                    == labview_data.loc[i_round + 1, "stripes_total"]
                )

                if (
                    labview_data.loc[i_round - 1, "stripes_per_round"]
                    > n_zones_expected - 1
                ):
                    labview_data.loc[i_round - 1, "stripes_per_round"] = (
                        n_zones_expected - 1
                    )
        return labview_data

    def _correct_belt_length(
        self,
        labview_data: pd.DataFrame,
        belt_length_mm: float = 1500.0,
        zone_lengths_mm: list = None,
    ):
        """
        Correct the belt length in the labview data.
        The belt length is not always correctly recorded in the LabView data.
        Args:
            labview_data (pd.DataFrame): The LabView data.
            belt_length_mm (float): The belt length in mm.
            zone_lengths_mm (list[float]): The number of zones (bordered by stripes) on the belt.
            If None, it is assumed that there are three zones of equal length.
        """
        n_rounds = np.diff(labview_data["round"]).sum()
        n_stripes = labview_data[
            "stripes_per_round"
        ].max()  # there is a bug in labview: the stripe counter
        # jumps to n_stripes + 1 in the last frame of one round before resetting to 0.
        # disregard this extra value (so do not add +1 to account for 0-indexing)
        if zone_lengths_mm is None:
            zone_lengths_mm = np.array(
                [belt_length_mm / n_stripes for i in range(n_stripes)]
            )
        if n_rounds < 1:  # no rounds recorded, so cannot correct to known length
            # TODO: is this correct? Should we not handle "last" round like below?
            return labview_data
        # use stripes per round and round to get each zone for each round.
        # if round 1 and zone 1 (it can be any zone (read out stripe_per_round value)):
        #       if distance at last frame smaller than expected:
        #           offset distance_PR so last entry matches expected zone distance.
        #       else:
        #           scale segment distance_PR by expected/actual,
        #           scale segment distance by expected/actual,
        #           offset distance for rest of dataset
        #               (- old first element post-segment + new last of segment)
        # for rest of rounds/zones (all rounds except last):
        #       offset distance_PR for zone by first element (to start at 0)
        #       scale to match expected zone length (factor = expected/actual)
        #       add offset distance_PR of last entry of last zone
        #       if not the first round:
        #          also correct cumulative distance (distance):
        #          offset distance for segment to start at 0, scale by factor expected/actual,
        #          add offset distance of last entry of last zone
        #  for last round:
        #       offset distance_PR and distance to last entry of last zone
        # TODO: extract repeated (offset + scale + offset again etc.) steps into a function
        for i_round in labview_data[
            "round"
        ].unique():  # this does not sort, but rounds should be sorted.
            mask_round = (
                labview_data["round"] == i_round
            )  # mask of all entries for this round
            idx_round = labview_data[mask_round].index
            if i_round == labview_data["round"].max():  # last round
                labview_data["distance_per_round"][idx_round] -= labview_data[
                    "distance_per_round"
                ][idx_round[0]]
                # last round, cannot match to expected length. Match beginning to end of previous
                # zone.
                labview_data["total_distance"][idx_round] = (
                    labview_data["total_distance"][idx_round]
                    - labview_data["total_distance"][idx_round[0]]
                    + labview_data["total_distance"][idx_round[0] - 1]
                )
                continue
            for i_zone_within_round, zone in enumerate(
                labview_data["stripes_per_round"][idx_round].unique()
            ):  # important: zone marks which zone we are in; i_zone_within_round = zone for all
                # zones except maybe the first (if the recording did not start in the first zone)
                if zone >= n_stripes:  # FIXME: remove this once correction is added
                    break
                # i_zone: always starts with 0; in first round, zone might start with >0 value!
                # FIXME: bug(?) in Matlab code: win = ...:stripe(i) includes beginning of next stripe...
                mask_zone = mask_round & (labview_data["stripes_per_round"] == zone)
                # FIXME: remove this, as this is a bug in the Matlab code
                # convert to [zone_start + 1 : zone_end+1], except for first zone, where it is
                # [zone_start : zone_end+1]
                mask_zone[np.where(mask_zone)[0][-1] + 1] = True
                i_first_frame_zone = np.where(mask_zone)[0][0]
                if i_first_frame_zone != 0 and i_zone_within_round != 0:
                    mask_zone[i_first_frame_zone] = False
                idx_zone = labview_data[mask_zone].index
                if (
                    i_round == 0 and i_zone_within_round == 0
                ):  # first round, first recorded zone within round
                    if (
                        labview_data["distance_per_round"][idx_zone].iloc[-1]
                        < zone_lengths_mm[i_zone_within_round]
                    ):
                        # offset distance_per_round so last entry matches expected zone
                        # distance
                        labview_data["distance_per_round"][idx_zone] = (
                            labview_data["distance_per_round"][idx_zone]
                            - labview_data["distance_per_round"][idx_zone].iloc[-1]
                            + zone_lengths_mm[zone]
                        )
                    else:
                        # scale segment distance_per_round by expected/actual
                        factor = (
                            np.sum(zone_lengths_mm[: zone + 1])
                            / labview_data["distance_per_round"][idx_zone].max()
                        )
                        labview_data["distance_per_round"][idx_zone] *= factor
                        # scale segment distance by expected/actual

                        # labview_data["total_distance"][idx_zone] = (
                        #    labview_data["distance_per_round"][idx_zone] * factor
                        # )  # FIXME: use distance instead of distance_per_round?
                        # FIXME: Matlab: the "factor" calculated is always 1 here, as it uses distancePR,
                        # which is scaled to the belt length in the previous line...
                        labview_data["total_distance"][idx_zone] = labview_data[
                            "distance_per_round"
                        ][
                            idx_zone
                        ]  # factor is always 1, see bug in Matlab code
                        # offset distance for rest of dataset
                        labview_data["total_distance"][idx_zone[-1] + 1 :] = (
                            labview_data["total_distance"][idx_zone[-1] + 1 :]
                            - labview_data["total_distance"][idx_zone[-1] + 1]
                            + labview_data["total_distance"][idx_zone[-1]]
                        )
                else:  # all other rounds/zones (except last, it is handled above)
                    # offset distance_per_round for zone by first element (to start at 0)
                    labview_data["distance_per_round"][idx_zone] -= labview_data[
                        "distance_per_round"
                    ][idx_zone[0]]
                    # scale to match expected zone length
                    # TODO: up to this point, it looks fine (same as matlab data in line 54)
                    factor = (
                        zone_lengths_mm[zone]
                        / labview_data["distance_per_round"][idx_zone].max()
                    )
                    labview_data["distance_per_round"][idx_zone] *= factor
                    # add offset distance_per_round of last entry of last zone
                    if (
                        i_zone_within_round > 0
                    ):  # new round -> starts from 0, not from last zone last entry
                        labview_data["distance_per_round"][idx_zone] += labview_data[
                            "distance_per_round"
                        ][idx_zone[0] - 1]
                    if i_round > 0:
                        # correct cumulative distance
                        # offset distance for segment to start at 0
                        labview_data["total_distance"][idx_zone] -= labview_data[
                            "total_distance"
                        ][idx_zone[0]]
                        # scale by factor expected/actual
                        factor = (
                            zone_lengths_mm[zone]
                            / labview_data["total_distance"][idx_zone].max()
                        )
                        labview_data["total_distance"][idx_zone] *= factor
                        # add offset distance of last entry of last zone
                        labview_data["total_distance"][idx_zone] += labview_data[
                            "total_distance"
                        ][idx_zone[0] - 1]
        return labview_data

    def _get_running_mask(
        self, labview_data: pd.DataFrame, threshold: float = 40.0, width: int = 250
    ) -> np.array:
        """
        Add a binary mask for running.
        Following the Matlab notation,
        Args:
            labview_data (pd.DataFrame): The LabView data.
            threshold (float): The threshold that the (smoothed) speed has to pass continuously
                (except when merging segments).
            width (int): The width of the time window between two "running" frames to merge them
                (fill the frames between them as "running").
        """
        running = np.zeros(len(labview_data))
        # find all values above threshold
        idx_above_threshold = matlab_smooth(labview_data["speed"].abs()) > threshold
        running[idx_above_threshold] = 1
        # FIXME: this follows Matlab convention, but that sets start of running one frame
        # before actual running == 1
        idx_running_start = np.where(np.diff(running) == 1)[0]
        # This finds last running == 1 in individual episodes. If data ends with episode,
        # it will not be included.
        idx_running_stop = np.where(np.diff(running) == -1)[0]
        # merge running episodes that are close to each other
        for i_break in range(min(len(idx_running_start) - 1, len(idx_running_stop))):
            # check window between next loco start and current loco end
            if idx_running_start[i_break + 1] - idx_running_stop[i_break] < width:
                running[
                    idx_running_stop[i_break] : idx_running_start[i_break + 1] + 1
                ] = (
                    1  # FIXME starting point marks the first 0 before 1, so need to include it
                    # when changing to 1
                )
        return running

    def _set_initial_values(self, lv_data: pd.DataFrame) -> pd.DataFrame:
        """
        Set initial values for the LabView data.
        Parameters where initial value is set to 0:
            round
            distance (total_distance)
            stripes (stripes_total)

        Args:
            lv_data (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        lv_data["round"] = lv_data["round"] - lv_data["round"].iloc[0]
        lv_data["total_distance"] = (
            lv_data["total_distance"] - lv_data["total_distance"].iloc[0]
        )
        lv_data["stripes_total"] = (
            lv_data["stripes_total"] - lv_data["stripes_total"].iloc[0]
        )
        return lv_data
