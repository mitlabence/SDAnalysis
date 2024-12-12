"""
linear_locomotion.py - Module to match LabView and ND2 time stamps to get linear motion data. 
"""

import warnings
import numpy as np
import pandas as pd
from nd2_time_stamps import ND2TimeStamps
from lv_data import LabViewData
from lv_time_stamps import LabViewTimeStamps


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
    ):
        # time stamps are in seconds by default; see @property functions for milliseconds
        self.scanner_time_stamps = (
            nd2_time_stamps.time_stamps
        )  # TODO: this will later become Series, so do not assign here? (misleading)
        self.time_column = nd2_time_stamps.time_column  # FIXME: should remove this once pipeline
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
        # start at 0
        self.lv_time_stamps_belt = (
            self.lv_time_stamps_belt - self.lv_time_stamps_belt.iloc[0]
        )
        # reset indices
        self.lv_data = self.lv_data.reset_index(drop=True)
        self.lv_time_stamps_belt = self.lv_time_stamps_belt.reset_index(drop=True)
        self.lv_time_stamps_scanner = self.lv_time_stamps_scanner.reset_index(drop=True)
        # replace time stamps in belt
        self.lv_data["time_total_s"] = self.lv_time_stamps_belt

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
