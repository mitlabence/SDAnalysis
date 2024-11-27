"""
lv_data.py - Module to read and process data from LabView-exported txt files.
"""
import os
import pandas as pd

class LabViewData:
    """
    Class to read and process data from LabView-exported txt files.
    """
    def __init__(self, file_path: str, encoding: str = "utf-8", separator: str = "\t"):
        self.file_path = file_path
        self.encoding = encoding
        self.separator = separator
        self.data = self._read_file()
    def _read_file(self):
        if self.file_path is None:
            raise ValueError("No file path provided")
        if not os.path.exists(self.file_path):
            raise ValueError("Invalid file path provided")
        df_result = pd.read_table(self.file_path, encoding=self.encoding, sep=self.separator, header=None)
        df_result = df_result.dropna(axis=0, how="any")  # Drop last row that is not complete
        return df_result