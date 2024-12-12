"""
lv_data.py - Module to read and process data from LabView-exported txt files.
"""

import os
from typing import List
import pandas as pd


class LabViewData:
    """
    Class to read and process data from LabView-exported txt files.
    Units of time are in seconds by default; see @property functions for milliseconds.
    """

    def __init__(self, file_path: str, encoding: str = "utf-8", separator: str = "\t"):
        self.file_path = file_path
        self.encoding = encoding
        self.separator = separator
        self.col_names = [
            "rounds",
            "speed",
            "total_distance",
            "distance_per_round",
            "reflectivity",
            "unknown",
            "stripes_total",
            "stripes_per_round",
            "time_total_s",
            "time_per_round",
            "stimuli1",
            "stimuli2",
            "stimuli3",
            "stimuli4",
            "stimuli5",
            "stimuli6",
            "stimuli7",
            "stimuli8",
            "stimuli9",
            "pupil_area",
        ]
        self.data = self._read_file_to_data_frame(col_names=self.col_names).reset_index(
            drop=True
        )  # At this point, time_total_s column (and time_per_round) is in milliseconds 
        # (as in labview file)
        # convert to seconds
        self.data = self._convert_to_s(self.data)

    @property
    def data_ms(self):
        """
        The labview data in milliseconds.
        """
        return self._convert_to_ms(self.data)

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
            original_dtype = df[col].dtype
            df[col] = df[col] * 1000.0
            df[col] = df[col].astype(original_dtype)
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
        df = df.copy()
        time_columns = [col for col in df.columns if "time" in col]
        for col in time_columns:
            df[col] = df[col] / 1000.0
        return df

    def _read_file_to_data_frame(self, col_names: List[str]) -> pd.DataFrame:
        """
        Read the dataset from a txt file and assign column names

        Args:
            col_names (_type_): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        if self.file_path is None:
            raise ValueError("No file path provided")
        if not os.path.exists(self.file_path):
            raise ValueError("Invalid file path provided")
        df_result = pd.read_table(
            self.file_path, encoding=self.encoding, sep=self.separator, header=None
        )
        df_result.columns = col_names
        #df_result = df_result.dropna(
        #    axis=0, how="any"
        #)  # Drop last row that is not complete; matlab keeps it so we should too!
        return df_result
