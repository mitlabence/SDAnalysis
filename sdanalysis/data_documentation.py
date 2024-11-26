"""
data_documentation.py - Functions to load and process the data documentation.
"""

import os
import warnings
from math import isnan
import pandas as pd
import numpy as np
import duckdb
import custom_io as cio


# TODO: reading from duckdb results in categorical values for many columns
# (for example, SEGMENTATION_DF.interval_type). This might lead to unexpected behavior.


class DataDocumentation:
    """
    segments_cnmf_cats and segments_moco_cats assign to each category appearing in the data
    documentation (segmentation) the boolean value whether the CNMF and MoCo can or should run
    on segments belonging in that category. These categories should be exactly the unique
    categories appearing in the [mouse-id]_segmentation.xlsx files, or, once segmentation_df
    contains all this data, in segmentation_df["interval_type"].unique().
    """

    _datadoc_path = None
    grouping_df = None  # df containing files belonging together in a session
    segmentation_df = None  # df containing segmentation
    colorings_df = None  # df containing color code for each mouse ID
    # df containing window side, type, injection side, type.
    win_inj_types_df = None
    events_df = None  # df containing all events and the corresponding metadata
    segments_cnmf_cats = {
        "normal": True,
        "iis": False,
        "sz": False,
        "sz_like": False,
        "sd_wave": False,
        "sd_extinction": False,
        "fake_handling": False,
        "sd_wave_delayed": False,
        "sd_extinction_delayed": False,
        "stimulation": False,
        "sd_wave_cx": False,
        "window_moved": False,
        "artifact": False,
    }
    segments_moco_cats = {
        "normal": True,
        "iis": True,
        "sz": True,
        "sz_like": True,
        "sd_wave": True,
        "sd_extinction": True,
        "fake_handling": True,
        "sd_wave_delayed": True,
        "sd_extinction_delayed": True,
        "stimulation": False,
        "sd_wave_cx": True,
        "window_moved": True,
        "artifact": False,
    }

    def __init__(self, datadoc_path: str = None):
        if datadoc_path is None:
            self._datadoc_path = cio.open_dir("Open data documentation")
        else:
            self._datadoc_path = datadoc_path
        # make sure either folder or duckdb file is given, and that it exists
        if os.path.isdir(self._datadoc_path) and not os.path.exists(self._datadoc_path):
            raise NotADirectoryError(f"{self._datadoc_path} is not a valid directory.")
        if (
            os.path.isfile(self._datadoc_path)
            and not os.path.splitext(self._datadoc_path)[-1] == ".duckdb"
        ):
            raise FileNotFoundError(f"{self._datadoc_path} is not a valid duckdb file.")

    @classmethod
    def from_env_dict(cls, env_dict):
        """Given .env as dict, open the data documentation file and return the DataDocumentation
        object.
        Parameters
        ----------
        env_dict : dict
            The dictionary read out of the project .env file (see env_reader.read_env())

        Returns
        -------
        DataDocumentation
            the data documentation
        """
        # Set up data documentation
        if "DATA_DOCU_FOLDER" in env_dict.keys():
            docu_folder = env_dict["DATA_DOCU_FOLDER"]
        else:
            docu_folder = cio.open_dir(
                "Choose folder containing folders for each mouse!"
            )
        # Load data documentation
        ddoc = DataDocumentation(docu_folder)
        ddoc._load_data_doc()
        return ddoc

    def check_category_consistency(self):
        """
        Checks the consistency of category segments between the segmentation dataframe
        and the predefined categories in SEGMENTS_CNMF_CATS and SEGMENTS_MOCO_CATS.

        Raises:
            ValueError: If the number of unique segment types in the segmentation dataframe
                        does not match the number of segment types defined in SEGMENTS_CNMF_CATS.
            ValueError: If the number of unique segment types in the segmentation dataframe
                        does not match the number of segment types defined in SEGMENTS_MOCO_CATS.

        Prints:
            A message indicating that the categories seem consistent if no inconsistencies
            are found.
        """
        n_segments = len(self.segmentation_df["interval_type"].unique())
        n_segments_cnmf = len(self.segments_cnmf_cats.keys())
        n_segments_moco = len(self.segments_moco_cats.keys())
        if n_segments != n_segments_cnmf:
            raise ValueError(
                f"Found {n_segments} segment types in data documentation: \
                {self.segmentation_df['interval_type'].unique()} vs {n_segments_cnmf} defined in \
                    datadoc_util.py (segments_cnmf_cats): {self.segments_cnmf_cats.keys()}"
            )
        if n_segments != n_segments_moco:
            raise ValueError(
                f"Found {n_segments} segment types in data documentation: \
                           {self.segmentation_df['interval_type'].unique()} vs {n_segments_cnmf} \
                            defined in datadoc_util.py (segments_moco_cats): \
                                {self.segments_moco_cats.keys()}"
            )
        print(
            "DataDocumentation.checkCategoryConsistency(): Categories seem consistent."
        )

    def _check_file_consistency(
        self,
    ):
        """
        Go over the sessions contained in the data documentation; print the files that were not
        found.
        :return:
        """
        count_not_found = 0
        for _, grouping_row in self.grouping_df.iterrows():
            folder = grouping_row["folder"]
            nd2_fname = grouping_row["nd2"]
            lv_fname = grouping_row["labview"]
            lfp_fname = grouping_row["lfp"]
            facecam_fname = grouping_row["face_cam_last"]
            nikonmeta_fname = grouping_row["nikon_meta"]
            for fname in [
                nd2_fname,
                lv_fname,
                lfp_fname,
                facecam_fname,
                nikonmeta_fname,
            ]:
                if isinstance(fname, str):
                    fpath_complete = os.path.join(folder, fname)
                    if not os.path.exists(fpath_complete):
                        print(f"Could not find {fpath_complete}")
                        count_not_found += 1
        print(f"Total: {count_not_found} missing files.")

    def _load_data_doc(self):
        # check of existance was done above in __init__
        if os.path.isfile(self._datadoc_path):
            self._load_from_file()
        elif os.path.isdir(self._datadoc_path):
            self._load_from_folder()

    @staticmethod
    def _uuid_to_string(df_to_format, column_name):
        """
        Given a dataframe and a column name, convert the UUID objects in the column to string.

        Args:
            df_to_format (_type_): _description_
            column_name (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if not column_name in df_to_format.columns:
            raise ValueError(f"Column {column_name} not found in dataframe.")
        return df_to_format[column_name].apply(
            lambda uuid: uuid if isnan(uuid) else uuid.hex
        )

    def _load_from_file(self):
        """Load the data documentation from a duckdb file. No check for file existence is done
        here, as it is done in __init__."""
        conn = duckdb.connect(self._datadoc_path)
        self.grouping_df = (
            conn.execute("SELECT * FROM grouping").fetchdf().fillna(np.NaN)
        )
        self.segmentation_df = (
            conn.execute("SELECT * FROM segmentation").fetchdf().fillna(np.NaN)
        )
        self.win_inj_types_df = (
            conn.execute("SELECT * FROM win_inj_types").fetchdf().fillna(np.NaN)
        )
        self.events_df = conn.execute("SELECT * FROM events").fetchdf().fillna(np.NaN)
        self.colorings_df = (
            conn.execute("SELECT * FROM colors").fetchdf().fillna(np.NaN)
        )
        # format uuid columns
        self.grouping_df["uuid"] = self._uuid_to_string(self.grouping_df, "uuid")
        self.events_df["event_uuid"] = self._uuid_to_string(
            self.events_df, "event_uuid"
        )
        self.events_df["recording_uuid"] = self._uuid_to_string(
            self.events_df, "recording_uuid"
        )

    def _load_from_folder(self):
        # reset the dataframes
        for root, _, files in os.walk(self._datadoc_path):
            for name in files:
                ext = os.path.splitext(name)[-1]
                if ext != ".xlsx":
                    continue
                if (
                    "~" in name
                ):  # "~" on windows is used for temporary files that are opened in excel
                    raise IOError(
                        f"Please close all excel files and try again. Found temporary file \
                            in:\n{os.path.join(root, name)}"
                    )
                if "grouping" in name:
                    df_readout = pd.read_excel(os.path.join(root, name))
                    mouse_id = os.path.splitext(name)[0].split("_")[
                        0
                    ]  # get rid of extension, then split xy_grouping to get xy
                    df_readout["mouse_id"] = mouse_id
                    self.grouping_df = (
                        df_readout
                        if self.grouping_df is None
                        else pd.concat([self.grouping_df, df_readout])
                    )
                elif "segmentation" in name:
                    df_readout = pd.read_excel(os.path.join(root, name))
                    self.segmentation_df = (
                        df_readout
                        if self.segmentation_df is None
                        else pd.concat([self.segmentation_df, df_readout])
                    )
                elif name == "window_injection_types_sides.xlsx":
                    self.win_inj_types_df = pd.read_excel(os.path.join(root, name))
                elif name == "events_list.xlsx":
                    self.events_df = pd.read_excel(os.path.join(root, name))
        if self.win_inj_types_df is None:
            raise ValueError(
                "Window_injection_types_sides.xlsx was not found in data documentation! \
            Possible reason is the changed structure of data documentation. This file was moved \
                out of 'documentation'. Do not move it back!"
            )
        self.colorings_df = self.get_colorings()
        # adjust data types to match duckdb types
        self.grouping_df.stim_length = self.grouping_df.stim_length.astype(
            np.float32
        )  # reduce float64 to 32
        self.segmentation_df.frame_begin = self.segmentation_df.frame_begin.astype(
            np.int32
        )
        self.segmentation_df.frame_end = self.segmentation_df.frame_end.astype(np.int32)

    def set_data_drive_symbol(self, symbol: str = None):
        """
        Set the symbol for the drive that appears in the file paths of the session file groupings.
        If no symbol is given, the drive will not be changed.

        Args:
            symbol (str, optional): The drive letter, i.e. C. Defaults to None.
        """
        if isinstance(symbol, str):
            assert len(symbol) == 1
            assert symbol.upper() == symbol
            self.grouping_df.folder = self.grouping_df.apply(
                lambda row: symbol + row["folder"][1:], axis=1
            )
            print(f"Changed drive symbol to {symbol}")

    def get_id_uuid(self):
        """
        Get the mouse ID and UUID columns for all UUID from the data documentation.

        Raises:
            SyntaxError: _description_

        Returns:
            _type_: _description_
        """
        if self.grouping_df is not None:
            return self.grouping_df[["mouse_id", "uuid"]]
        raise SyntaxError(
            "datadoc_util.DataDocumentation.getIdUuid: You need to run loadDataDoc() first to \
                populate DataDocumentation object"
        )

    def get_colorings(self):
        """
        Read out the 'color coding.xlsx' of the data documentation, which should contain ID -
         color hex, r, g, b pairs.
        :return: pandas dataframe
        """
        color_coding_fpath = os.path.join(self._datadoc_path, "color coding.xlsx")
        if os.path.exists(color_coding_fpath):
            return pd.read_excel(color_coding_fpath)
        raise FileNotFoundError(f"File {color_coding_fpath} does not exist.")

    def get_mouse_id_for_uuid(self, uuid):
        """
        Given a UUID, return the mouse ID. If the UUID does not exist, an error is raised.
        """
        # TODO: handle invalid uuid
        return self.grouping_df[self.grouping_df["uuid"] == uuid].mouse_id.values[0]

    def get_mouse_win_inj_info(self, mouse_id):
        """
        Given a mouse ID, return the window and injection information for that mouse.
        If the mouse ID is not found, an error is raised.

        Args:
            mouse_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.win_inj_types_df[self.win_inj_types_df["mouse_id"] == mouse_id]

    def get_injection_direction(self, mouse_id):
        """
        Given a mouse ID, return the direction of the injection. If the mouse ID is not found,
        an error is raised.

        Args:
            mouse_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        inj_side = self.win_inj_types_df[self.win_inj_types_df["mouse_id"] == mouse_id][
            "injection_side"
        ].values[0]
        win_side = self.win_inj_types_df[self.win_inj_types_df["mouse_id"] == mouse_id][
            "window_side"
        ].values[0]
        top_dir = self.get_top_direction(mouse_id)
        # assert the injection side is opposite to the window side
        if (inj_side == "right" and win_side == "left") or (
            inj_side == "left" and win_side == "right"
        ):
            # if imaging contralateral to injection, injection is always towards medial side
            return "top" if top_dir == "medial" else "bottom"
        return "same"

    def _get_grouping(self, fpath):
        """
        Given a file path, return the grouping information for that file. (What other files
        belong to the same session)

        Args:
            fpath (_type_): _description_

        Raises:
            FileExistsError: _description_
            FileExistsError: _description_
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        fname = os.path.split(fpath)[-1]
        ftype = os.path.splitext(fpath)[-1]
        if ftype == ".nd2":
            df_grouping_results = self.grouping_df[self.grouping_df["nd2"] == fname]
            if len(df_grouping_results) > 1:
                raise FileExistsError("nd2 file is not unique!")
        elif ftype == ".abf":
            df_grouping_results = self.grouping_df[self.grouping_df["lfp"] == fname]
            if len(df_grouping_results) > 1:
                raise FileExistsError("LFP file is not unique!")
        else:
            raise NotImplementedError(
                f"Function not yet implemented for files of type {ftype}!"
            )
        return df_grouping_results

    def get_uuid_for_file(self, fpath):
        """
        Retrieve the UUID for a given file path.

        This method attempts to get the UUID associated with the specified file path
        by querying a DataFrame obtained from the `_get_grouping` method. If the UUID
        is not found, a warning is issued and None is returned.

        Parameters:
        -----------
        fpath : str
            The file path for which to retrieve the UUID.

        Returns:
        --------
        str or None
            The UUID associated with the file path, or None if not found.
        """
        df_query_result = self._get_grouping(fpath)
        try:
            return df_query_result["uuid"].iat[0]
        except IndexError:
            warnings.warn(f"Could not find uuid for {fpath}")
            return None

    def get_experiment_type_for_file(self, fpath: str):
        """Given a file path, determine the experiment type of that session. If the experiment
        type is not found, a ValueError is raised.

        Args:
            fpath (str): The file path

        Returns:
            str: The experiment type
        """
        df_query_results = self._get_grouping(fpath)
        try:
            return df_query_results["experiment_type"].iat[0]
        except IndexError as exc:
            raise ValueError(f"Could not find experiment type for {fpath}") from exc

    def get_segments(self, nd2_file):
        """
        Given an nd2 file name, return the segments for that file. If the file is not found
        in the documentation, an empty dataframe is returned.

        Args:
            nd2_file (_type_): _description_

        Returns:
            pd.DataFrame: _description_
        """
        assert os.path.splitext(nd2_file)[-1] == ".nd2"
        return self.segmentation_df[self.segmentation_df["nd2"] == nd2_file]

    def get_segments_for_uuid(self, uuid, as_df=True):
        """
        Given a UUID, return the segments for that recording. If as_df is True, return the
        segments as a dataframe. Otherwise, return the segments as a list of tuples, where each
        tuple contains the interval type, the beginning frame, and the end frame of the segment.
        Args:
            uuid (_type_): _description_
            as_df (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        nd2_file = self.grouping_df[self.grouping_df["uuid"] == uuid].nd2.values[0]
        segments_df = self.segmentation_df[self.segmentation_df["nd2"] == nd2_file]
        segments_df = segments_df.drop("nd2", axis=1)
        if as_df:
            return segments_df
        ival_type = segments_df["interval_type"].array
        fbegin = segments_df["frame_begin"].array
        fend = segments_df["frame_end"].array
        segments = []
        for i, ival_type_item in enumerate(ival_type):
            segments.append((ival_type_item, fbegin[i], fend[i]))
        return segments

    def get_top_direction(self, mouse_id):
        """
        Given a mouse ID, return the top direction of the imaging field. If the mouse ID is not
        found, an error is raised.

        Args:
            mouse_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        # push right button: mouse goes forward, imaging field goes right (cells go left).
        # This means that
        # right: posterior, left: anterior.
        # Depending on window side:
        # left window: up=towards medial, down: towards lateral
        # right window: up=towards lateral, down: towards medial
        window_side_top_dir_dict = {"left": "medial", "right": "lateral"}
        return window_side_top_dir_dict[
            self.get_mouse_win_inj_info(mouse_id)["window_side"].values[0]
        ]

    def _add_uuid_column_from_nd2(self, df_query_result):
        """
        Purpose: when returning dataframe, useful to add uuid column instead of only specifying
        nd2 files. This function adds the uuid column to a dataframe (e.g. self.SEGMENTATION_DF)
        :param df:
        :return:
        """
        assert "nd2" in df_query_result.columns
        if "uuid" not in df_query_result.columns:
            df_query_result["uuid"] = df_query_result.apply(
                lambda row: self.grouping_df[
                    self.grouping_df["nd2"] == row["nd2"]
                ].uuid.values[0],
                axis=1,
            )
        else:
            print(
                "datadoc_util addUUIDColumnFromNd2: uuid column already exists! Returning \
                    unchanged dataframe..."
            )
        return df_query_result

    def get_all_segments_with_type(
        self, segment_type="normal", experiment_type: str = "tmev"
    ):
        """
        Returns all segments in the data documentation that have the defined segment type(s)
        :param segment_type: string or list of strings.
        :param experiment_type: string or list of strings. only take recordings that fall into
        this category (see data grouping). Example: "tmev", "tmev_bl", "chr2_szsd"
        :return:
        """

        # convert possible string
        if isinstance(segment_type, str):
            segment_types = [segment_type]
        else:  # assume otherwise list of strings was given
            segment_types = segment_type

        if isinstance(experiment_type, str):
            experiment_types = [experiment_type]
        else:
            experiment_types = experiment_type
        exptype_unique = self.grouping_df.experiment_type.unique()
        for e_type in experiment_types:  # cannot do for element in experiment_types
            assert e_type in exptype_unique

        res_df = self._add_uuid_column_from_nd2(
            self.segmentation_df[
                self.segmentation_df["interval_type"].isin(segment_types)
            ]
        )
        res_df["experiment_type"] = res_df.apply(
            lambda row: self.grouping_df[
                self.grouping_df["nd2"] == row["nd2"]
            ].experiment_type.values[0],
            axis=1,
        )
        res_df = res_df[res_df["experiment_type"].isin(experiment_types)]
        return res_df  # .drop("experiment_type")

    def get_nikon_file_name_and_uuid(self):
        """
        Get the nd2 file name and the UUID for all UUID from the data documentation.

        Raises:
            SyntaxError: _description_

        Returns:
            _type_: _description_
        """
        if self.grouping_df is not None:
            return self.grouping_df[["nd2", "uuid"]]
        raise SyntaxError(
            "datadoc_util.DataDocumentation.getIdUuid: You need to run loadDataDoc() first \
                to populate DataDocumentation object"
        )

    def get_nikon_file_name_for_uuid(self, uuid):
        """
        Given a recording uuid or a list of recording uuids, get the corresponding nikon recording
        (if it exists). If any UUID does not have a recording, an error is raised.

        Args:
            uuid (_type_): _description_

        Raises:
            TypeError: _description_
            SyntaxError: _description_

        Returns:
            _type_: _description_
        """
        if self.grouping_df is not None:
            if isinstance(uuid, str):
                return self.grouping_df[self.grouping_df["uuid"] == uuid].nd2.values[0]
            if isinstance(uuid, list):
                return self.grouping_df[self.grouping_df["uuid"].isin(uuid)].nd2.values
            raise TypeError(
                f"uuid has type {type(uuid)}; needs to be str or list[str]!"
            )
        raise SyntaxError(
            "datadoc_util.DataDocumentation.getIdUuid: You need to run loadDataDoc() first to "
            "populate DataDocumentation object"
        )

    def get_nikon_file_path_for_uuid(self, uuid):
        """
        Given a recording uuid or a list of recording uuids, get the corresponding nikon recording
        (if it exists). If any UUID does not have a recording, an error is raised.

        Args:
            uuid (str or List[str]): _description_

        Raises:
            TypeError: _description_
            SyntaxError: _description_

        Returns:
            _type_: _description_
        """
        if self.grouping_df is not None:
            if isinstance(uuid, str):
                grouping_entry = self.grouping_df[
                    self.grouping_df["uuid"] == uuid
                ].iloc[0]
                folder = grouping_entry.folder
                nd2 = grouping_entry.nd2
                return os.path.join(folder, nd2)
            if isinstance(uuid, (list, np.ndarray)):
                fpath_list = []
                for uuid_entry in uuid:
                    entry = self.grouping_df[
                        self.grouping_df["uuid"] == uuid_entry
                    ].iloc[0]
                    folder = entry.folder
                    nd2 = entry.nd2
                    fpath_list.append(os.path.join(folder, nd2))
                return fpath_list
            raise TypeError(
                f"uuid has type {type(uuid)}; needs to be str or list[str]!"
            )
        raise SyntaxError(
            "DataDocumentation.getIdUuid: You need to run loadDataDoc() first to populate "
            "DataDocumentation object"
        )

    def get_session_files_for_uuid(self, uuid):
        """
        Given a recording uuid, return the file paths for all files in the session. If the UUID
        is not found, an error is raised.

        Args:
            uuid (_type_): _description_

        Raises:
            SyntaxError: _description_

        Returns:
            _type_: _description_
        """
        if self.grouping_df is not None:
            return self.grouping_df[self.grouping_df["uuid"] == uuid]
        raise SyntaxError(
            "DataDocumentation.getIdUuid: You need to run loadDataDoc() first to populate "
            "DataDocumentation object"
        )

    def get_segment_for_frame(self, uuid, frame):
        """
        Given a 1-indexed frame, return a row containing interval type, beginning frame, end frame
        for the segment that the frame belongs to.
        :param uuid: The uuid of the recording.
        :param frame: The 1-indexed frame (i.e. first frame = 1) to get the segment info on.
        :return: a pandas DataFrame with columns "nd2", "interval_type", "frame_begin", "frame_end",
        and with at most one row, the segment that the frame belongs to.
        """
        nd2_file = self.grouping_df[self.grouping_df["uuid"] == uuid].nd2.values[0]
        return self.segmentation_df[
            (self.segmentation_df["nd2"] == nd2_file)
            & (self.segmentation_df["frame_begin"] <= frame)
            & (self.segmentation_df["frame_end"] >= frame)
        ]

    def get_color_for_mouse_id(self, mouse_id):
        """
        Get the color code for a mouse ID.

        Args:
            mouse_id (_type_): _description_

        Raises:
            ValueError: _description_
            SyntaxError: _description_

        Returns:
            _type_: _description_
        """
        if self.colorings_df is not None:
            if mouse_id in self.colorings_df.mouse_id.unique():
                return (
                    self.colorings_df[self.colorings_df["mouse_id"] == mouse_id]
                    .iloc[0]
                    .color
                )
            raise ValueError(f"Color code for ID {mouse_id} not found")
        raise SyntaxError("Color codes not yet loaded.")

    def get_color_for_uuid(self, uuid):
        """
        Given a recording uuid (not an event uuid), return the color code for the mouse ID of
        the recording.

        Args:
            uuid (_type_): _description_

        Returns:
            _type_: _description_
        """
        mouse_id = self.get_mouse_id_for_uuid(uuid)
        color = self.get_color_for_mouse_id(mouse_id)
        return color

    def get_recordings_with_experiment_type(self, experiment_types="fov_dual"):
        """
        Return all grouping info of recordings with the defined experiment_type,
        :param experiment_types: string or list of strings, type(s) of experiment. Some examples:
        fov_dual, tmev, tmev_bl, chr2_szsd, chr2_ctl, chr2_sd
        :return:
        """
        if isinstance(experiment_types, str):
            return self.grouping_df[
                self.grouping_df.experiment_type == experiment_types
            ]
        if isinstance(experiment_types, list):
            assert isinstance(experiment_types[0], str)
            return self.grouping_df[
                self.grouping_df.experiment_type.isin(experiment_types)
            ]
        raise TypeError("experiment_types must be str or list of str")

    def get_events_df(self):
        """
        Get the (TMEV) events dataframe from the data documentation. Events are combination of
        (TMEV) recordings that make up a baseline, a seizure(+SD), and a postictal phase.

        Returns:
            pandas.DataFrame: the events dataframe
        """
        return self.events_df

    def get_experiment_type_for_uuid(self, uuid):
        """
        Given a uuid, return the experiment type of the recording (or an empty dataframe if
        not found).

        Args:
            uuid (str): The recording uuid. Must not be event uuid (i.e. the uuid of a dataset
            that consists of multiple recordings).

        Returns:
            pd.DataFrame: _description_
        """
        return self.grouping_df[self.grouping_df["uuid"] == uuid].experiment_type.iloc[
            0
        ]

    def get_stim_duration_for_uuid(self, uuid):
        """
        Return the stim duration in seconds.
        """
        return self.grouping_df[self.grouping_df["uuid"] == uuid].stim_length.iloc[0]

    def get_events_list(self) -> pd.DataFrame:
        """
        Get the events list from the data documentation.

        Returns:
            pd.DataFrame: _description_
        """
        return self.events_df
