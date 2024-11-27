"""
nd2_time_stamps.py - Module to read and process time stamps from nd2 files.
"""

import os
from typing import List
import pandas as pd


class ND2TimeStamps:
    """
    Class to read and process time stamps from nd2 files.
    """

    def __init__(self, file_path, encoding: str = "utf-16", separator: str = "\t"):
        self.file_path = file_path
        self.encoding = encoding
        self.separator = separator
        time_stamps = self._read_out_file()
        time_stamps = self._strip_split_lines(time_stamps)
        if self._is_header_like(time_stamps[0]):
            self.header, time_stamps = time_stamps[0], time_stamps[1:]
        time_stamps = self._unify_decimal_separators(time_stamps)
        time_stamps = self._find_format_hourlike(time_stamps)
        time_stamps = self._convert_string_to_numeric(time_stamps)
        self.i_index_column = self._find_index_column()
        self.special_frames, time_stamps = self._extract_special_frames(
            time_stamps, self.i_index_column
        )
        self.time_stamps = pd.DataFrame(
            time_stamps, columns=self.header[: len(time_stamps[0])]
        )

    def _read_out_file(self):
        """Try to open contents of the nd2 time stamps file

        Raises:
            ValueError: If no path is provided or it is invalid

        Returns:
            _type_: _description_
        """
        if self.file_path is None:
            raise ValueError("No file path provided")
        if not os.path.exists(self.file_path):
            raise ValueError("Invalid file path provided")
        with open(self.file_path, "r", encoding=self.encoding) as time_stamps_file:
            time_stamps = time_stamps_file.readlines()
        return time_stamps

    def _find_index_column(self) -> int:
        if self.header is None:
            raise ValueError("No header found")
        for i_entry, entry in enumerate(self.header):
            if "index" in entry.lower():
                return i_entry
        raise ValueError("No index column found")

    @staticmethod
    def _is_header_like(line: List[str]) -> bool:
        """Check if the line is a header

        Args:
            line (List[str]): Header candidate

        Returns:
            bool: True if line is a header, False otherwise
        """
        return any(any(c.isalpha() for c in entry.lower()) for entry in line)

    def _strip_split_lines(self, lines: List[str]) -> List[str]:
        """
        Split stripped lines by self.separator

        Args:
            lines (List[str]): _description_

        Returns:
            List[str]: _description_
        """
        formatted_lines = []
        for line in lines:
            line = line.strip().split(self.separator)
            if line:
                formatted_lines.append(line)
        return formatted_lines

    def _find_format_hourlike(self, lines: List[str]) -> List[str]:
        """
        Find "hour-like" column and convert it to seconds. A "hour-like" column has a form like:
        mm:ss.ms. If none of the columns is like that, the returned list is unchanged.
        Args:
            lines (_type_): _description_
        """
        for i_line, line in enumerate(lines):
            for i_entry, entry in enumerate(line):
                if ":" in entry:
                    lines[i_line][i_entry] = self._format_hourlike_entry(entry)
        return lines

    @staticmethod
    def _format_hourlike_entry(entry: str):
        """
        Convert "hour-like" string entry to seconds. A "hour-like" entry has a form like:
        mm:ss.ms.

        Args:
            entry (str): _description_
        """
        if not isinstance(entry, str):
            raise ValueError("Entry is not a string")
        try:
            # assume mm:ss.ms format
            minutes, sec_ms = entry.split(":")
            minutes, sec_ms = int(minutes), float(sec_ms)
            entry = minutes * 60 + sec_ms
            return entry
        except ValueError as exc:
            raise ValueError(f"Invalid hour-like format: {entry}") from exc

    def _unify_decimal_separators(self, lines: List[str]) -> List[str]:
        """
        Find and convert all decimal separators to ".". If an entry is not a string, it is left
        unchanged.

        Args:
            lines (List[str]): _description_

        Returns:
            List[str]: _description_
        """
        for i_line, line in enumerate(lines):
            for i_entry, entry in enumerate(line):
                if isinstance(entry, str) and "," in entry:
                    lines[i_line][i_entry] = entry.replace(",", ".")
        return lines

    def _convert_string_to_numeric(self, lines: List[str]) -> List[str]:
        """
        Convert string entries to int or float. If an entry is not a numeric string,
        it is left unchanged.

        Args:
            lines (List[str]): _description_

        Returns:
            List[str]: _description_
        """
        for i_line, line in enumerate(lines):
            for i_entry, entry in enumerate(line):
                if isinstance(entry, str):
                    if "." in entry:
                        lines[i_line][i_entry] = float(entry)
                    else:
                        try:
                            lines[i_line][i_entry] = int(entry)
                        except ValueError:
                            pass
        return lines

    def _extract_special_frames(self, lines, i_index_column):
        """
        Extract special frames from the time stamps. Special frames are those with no index.

        Args:
            lines (List[List[Any]]): List of read-out lines
            i_index_column (int): the index (starting with 0) of the index column (the column
            with the frame number)
        """
        special_frames = []
        for i_line, line in enumerate(lines):
            if not line[i_index_column]:
                special_frames.append(line)
                lines.pop(i_line)
        return special_frames, lines