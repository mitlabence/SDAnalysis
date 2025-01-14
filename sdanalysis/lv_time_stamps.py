"""
lv_time_stamps.py - Module to read and process time stamps from LabView-exported txt time stamp files.
"""

import os
import warnings
from typing import List
import pandas as pd


class LabViewTimeStamps:
    """
    Class to read and process time stamps from LabView-exported txt time stamp files.
    All time stamps are in seconds by default; see @property functions for milliseconds.
    """

    def __init__(self, file_path: str, encoding: str = "utf-8", separator: str = "\t"):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.encoding = encoding
        self.separator = separator
        self.col_names = [
            "belt_time_stamps",
            "resonant_time_stamps",
            "galvano_time_stamps",
            "unused",
        ]
        self.time_stamps: pd.DataFrame = self._read_file(col_names=self.col_names)
        # file contains time stamps in ms, convert to s
        self.time_stamps = self._convert_to_s(self.time_stamps)
        self.recording_mode = self._get_recording_mode()
        self.scanner_time_stamps = (
            self.time_stamps["resonant_time_stamps"][
                self.time_stamps["resonant_time_stamps"] > 0
            ]
            if self.recording_mode == "resonant"
            else self.time_stamps["galvano_time_stamps"][
                self.time_stamps["galvano_time_stamps"] > 0
            ]
        ).reset_index(drop=True)

    @property
    def scanner_time_stamps_ms(self):
        """
        The labview time stamps of the scanner in milliseconds.
        """
        return self.scanner_time_stamps * 1000.0

    @property
    def time_stamps_ms(self):
        """
        The labview time stamps in milliseconds.
        """
        return self._convert_to_ms(self.time_stamps)

    def _convert_to_ms(self, df: pd.DataFrame):
        """
        Convert all time-related columns to milliseconds.

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = df.copy()
        time_columns = [col for col in df.columns if "time" in col]
        for col in time_columns:
            df[col] = df[col] * 1000.0
        return df

    def _convert_to_s(self, df: pd.DataFrame):
        """
        Find time-related columns, and convert them to seconds.

        Args:
            df (pd.DataFrame): _description_
        """
        df = df.copy()
        time_columns = [col for col in df.columns if "time" in col]
        for col in time_columns:
            df[col] = df[col] / 1000.0
        return df

    def _read_file(self, col_names: List[str]) -> pd.DataFrame:
        if self.file_path is None:
            raise ValueError("No file path provided")
        if not os.path.exists(self.file_path):
            raise ValueError("Invalid file path provided")
        df_result = pd.read_table(
            self.file_path, encoding=self.encoding, sep=self.separator, header=None
        )
        df_result.columns = col_names[: len(df_result.columns)]
        return df_result

    def _get_recording_mode(self):
        """
        Get the recording mode from the time stamps file: "resonant" or "galvano". The text file
        has two columns, 2 and 3, for each.
        """
        n_reso_tstamps = (self.time_stamps["resonant_time_stamps"] > 0).sum()
        n_galvano_tstamps = (self.time_stamps["galvano_time_stamps"] > 0).sum()
        if n_reso_tstamps * n_galvano_tstamps != 0:
            warnings.warn(
                f"Both resonant and galvano time stamps found: {n_reso_tstamps} \
                             resonant, {n_galvano_tstamps} galvano"
            )
        if n_reso_tstamps == 0 and n_galvano_tstamps == 0:
            raise ValueError("No resonant or galvano time stamps found")
        return "resonant" if n_reso_tstamps > n_galvano_tstamps else "galvano"
