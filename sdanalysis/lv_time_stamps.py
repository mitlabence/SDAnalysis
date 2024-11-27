"""
lv_time_stamps.py - Module to read and process time stamps from LabView-exported txt time stamp files.
"""

import os
import warnings
import pandas as pd


class LabViewTimeStamps:
    """
    Class to read and process time stamps from LabView-exported txt time stamp files.
    """

    def __init__(self, file_path: str, encoding: str = "utf-8", separator: str = "\t"):
        self.file_path = file_path
        self.encoding = encoding
        self.separator = separator
        self.time_stamps = self._read_file()
        self.recording_mode = self._get_recording_mode()

    def _read_file(self):
        if self.file_path is None:
            raise ValueError("No file path provided")
        if not os.path.exists(self.file_path):
            raise ValueError("Invalid file path provided")
        return pd.read_table(
            self.file_path, encoding=self.encoding, sep=self.separator, header=None
        )

    def _get_recording_mode(self):
        """
        Get the recording mode from the time stamps file: "resonant" or "galvano". The text file
        has two columns, 2 and 3, for each.
        """
        n_reso_tstamps = (self.time_stamps[1] > 0).sum()
        n_galvano_tstamps = (self.time_stamps[2] > 0).sum()
        if n_reso_tstamps * n_galvano_tstamps != 0:
            warnings.warn(
                f"Both resonant and galvano time stamps found: {n_reso_tstamps} \
                             resonant, {n_galvano_tstamps} galvano"
            )
        if n_reso_tstamps == 0 and n_galvano_tstamps == 0:
            raise ValueError("No resonant or galvano time stamps found")
        return "resonant" if n_reso_tstamps > n_galvano_tstamps else "galvano"
