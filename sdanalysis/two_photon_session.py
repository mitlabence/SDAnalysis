"""
two_photon_session.py - A class for opening and handling data of two-photon data sessions.
"""

import os
from typing import List
import datetime
import json
from copy import deepcopy
import warnings
from tkinter.filedialog import askopenfilename
import pyabf as abf  # https://pypi.org/project/pyabf/
import pims_nd2
import pandas as pd
import pytz  # timezones
import numpy as np
import nikon_ts_reader as ntsr
import h5py
from custom_io import open_dir, get_filename_with_date
from matplotlib import pyplot as plt
import scipy
from linear_locomotion import LinearLocomotion
from nd2_time_stamps import ND2TimeStamps
from lv_data import LabViewData
from lv_time_stamps import LabViewTimeStamps
import constants
from past.utils import old_div
import matplotlib as mpl
from scipy.sparse import spdiags
from caiman.utils.visualization import get_contours

try:
    import bokeh
    import bokeh.plotting as bpl
    from bokeh.models import (
        CustomJS,
        ColumnDataSource,
        Range1d,
        LabelSet,
        Dropdown,
        Slider,
    )
    from bokeh.models.widgets.buttons import Button, Toggle
    from bokeh.layouts import layout, row, column
except:
    print(
        "Bokeh could not be loaded. Either it is not installed or you are not running within \
        a notebook"
    )
import data_documentation as ddu  # import TwoPhotonSession from 2p-py folder (default use case of 2p-py). Otherwise absolute path here necessary!

# heuristic value, hopefully valid for all recordings made with the digitizer module
LFP_SCALING_FACTOR = 1.0038
# used for saving time stamps. See
# https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
# for further information. %f is 0-padded microseconds to 6 digits.
DATETIME_FORMAT = "%Y.%m.%d-%H:%M:%S.%f_%z"

# the column names of the labview xy.txt files
LV_COLNAMES = [
    "rounds",
    "speed",
    "total_distance",
    "distance_per_round",
    "reflectivity",
    "unknown",
    "stripes_total",
    "stripes_per_round",
    "time_total_ms",
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


# TODO: save to hd5 and open from hd5! Everything except the nikon movie could be saved
#    (and the dataframes). Logic is that we would not need the files anymore, we could combine
#   with caiman results. Or, if we want, we can open the nd2 file.
# TODO: in init_and_process, make functions that convert belt_dict and belt_scn_dict from matlab
#   data to numpy class methods that check if variable to convert is matlab array, if not,
#   does nothing.
# TODO: split up methods into more functions that are easily testable (like getting datetime
#   format from string using DATETIME_FORMAT) and write tests. Example: test various datetime
#   inputs for reading out from json (what if no timezone is supplied?)
# TODO: make export_json more verbose! (print save directory and file name, for example)
# TODO: make _match_lfp_nikon_stamps to instead of returning time_offs_lfp_nik change
#   self.time_offs_lfp_nik, like _create_nikon_daq_time. Anything against that?
# TODO: also make sure that each function that infers internal parameters can be run several
#   times and not mess up things, i.e. that parameters read are not changed.
# TODO: class private vs. module private (__ vs _):
#  https://stackoverflow.com/questions/1547145/defining-private-module-functions-in-python
# TODO: make inner functions (now _) class private, and not take any arguments. Just document
#   what attributes it reads and what attributes it changes/sets!
# TODO: add test that attributes of twophotonsession opened from json results file match those
#   of using init or init_and_process
# TODO: implement verbose flag to show/hide print() comments.


class TwoPhotonSession:
    """
    Attributes:
        Basic:
            nd2_path: str
            nd2_timestamps_path: str
            labview_path: str
            labview_timestamps_path: str
            lfp_path: str
            matlab_2p_folder: str

        Implied (assigned using basic attributes)
            nikon_movie
            nikon_true_length: int
            nikon_meta
            lfp_file: pyabf.abf.ABF, the raw lfp data
            belt_dict
            belt_scn_dict
            belt_df
            belt_scn_df
            nikon_daq_time: pandas.core.series.Series
            lfp_df
            lfp_df_cut
            time_offs_lfp_nik
            belt_params: dict
            lfp_t_start: datetime.datetime
            nik_t_start: datetime.datetime
            lfp_scaling: float = LFP_SCALING_FACTOR defined in __init__
            mean_fluo: np.array(np.float64?) mean fluorescence [optional]
            df_stim: pd.DataFrame the precise readout of the Nikon stimulation time stamps
                from the metadata file (i.e. the two rows stimulation begin and stimulation end)
    """

    # IMPORTANT: change the used lfp_scaling to always be the actual scaling used!
    # Important for exporting (saving) the session data for reproducibility

    def __init__(
        self,
        nd2_path: str = None,
        nd2_timestamps_path: str = None,
        labview_path: str = None,
        labview_timestamps_path: str = None,
        lfp_path: str = None,
        matlab_2p_folder: str = None,
        uuid: str = None,
        **kwargs,
    ):
        """
        Instantiate a TwoPhotonSession object with only basic parameters defined. This is the
        default constructor as preprocessing might take some time, and it might diverge for various
        use cases in the future.
        :param nd2_path: complete path of nd2 file
        :param nd2_timestamps_path: complete path of nd2 time stamps (txt) file
        :param labview_path: path of labview txt file (e.g. M278.20221028.133021.txt)
        :param labview_timestamps_path: complete path of labview time stamps
        (e.g. M278.20221028.133021time.txt)
        :param lfp_path: complete path of lfp (abf) file
        :param matlab_2p_folder: folder of matlab scripts (e.g. C:/matlab-2p/)
        :param kwargs: the inferred attributes of TwoPhotonSession can be directly supplied as keyword arguments. Useful
        e.g. for alternative constructors (like building class from saved json file)
        :return: None
        """
        # set basic attributes, possibly to None.
        self.nd2_path = nd2_path
        self.nd2_timestamps_path = nd2_timestamps_path
        self.labview_path = labview_path
        self.labview_timestamps_path = labview_timestamps_path
        self.lfp_path = lfp_path
        self.matlab_2p_folder = matlab_2p_folder
        self.uuid = uuid
        # set inferred attributes to default value
        self.nikon_movie = None
        self.nikon_meta = None
        self.lfp_file = None
        self.belt_dict = None
        self.belt_scn_dict = None
        self.belt_df = None
        self.belt_scn_df = None
        self.nikon_daq_time = None
        self.lfp_df = None
        self.lfp_df_cut = None
        self.time_offs_lfp_nik = None
        self.belt_params = None
        self.lfp_t_start = None
        self.nik_t_start = None
        self.lfp_scaling = None
        self.mean_fluo = None
        self.nikon_true_length = None
        self.verbose = True  # printing some extra text by default
        self.df_stim = None
        # in many stimulation recordings, the stimulation stamps in _nik file are off
        self.corrupted_stim_data = False

        # check for optionally supported keyword arguments:
        self._assign_from_kwargs("nikon_movie", kwargs)
        self._assign_from_kwargs("nikon_meta", kwargs)
        self._assign_from_kwargs("lfp_file", kwargs)
        self._assign_from_kwargs("belt_dict", kwargs)
        self._assign_from_kwargs("belt_scn_dict", kwargs)
        self._assign_from_kwargs("belt_df", kwargs)
        self._assign_from_kwargs("belt_scn_df", kwargs)
        self._assign_from_kwargs("nikon_daq_time", kwargs)
        self._assign_from_kwargs("lfp_df", kwargs)
        self._assign_from_kwargs("lfp_df_cut", kwargs)
        self._assign_from_kwargs("time_offs_lfp_nik", kwargs)
        self._assign_from_kwargs("belt_params", kwargs)
        self._assign_from_kwargs("lfp_t_start", kwargs)
        self._assign_from_kwargs("nik_t_start", kwargs)
        self._assign_from_kwargs("lfp_scaling", kwargs)
        self._assign_from_kwargs("nikon_true_length", kwargs)
        self._assign_from_kwargs("df_stim", kwargs)

    def _assign_from_kwargs(self, attribute_name: str, kwargs_dict: dict):
        if attribute_name in kwargs_dict.keys():
            setattr(self, attribute_name, kwargs_dict[attribute_name])
        else:
            setattr(self, attribute_name, None)

    @classmethod
    def init_and_process_uuid(cls, uuid: str = None, matlab_2p_folder: str = None):
        """
        Instantiate a TwoPhotonSession object by defining the uuid and perform the processing steps automatically.
        The files will be located using the 2p-py/.env file, DATA_DOCU_FOLDER entry.
        :param uuid: The hexadecimal representation of the uuid as string. Example: "04b8cfbfa1c347058bb139b4661edcf1"
        :param matlab_2p_folder: folder of matlab scripts (e.g. C:/matlab-2p/). If None, the 2p-py/.env file matlab_2p_folder will be used.
        :return: The two photon session instance.
        """
        env_dict = dict()
        if not os.path.exists("./.env"):
            raise FileNotFoundError(".env does not exist")
        else:
            with open("./.env", "r", encoding="utf-8") as f:
                for line in f.readlines():
                    l = line.rstrip().split("=")
                    env_dict[l[0]] = l[1]
        if "DATA_DOCU_FOLDER" in env_dict.keys():
            data_docu_folder = env_dict["DATA_DOCU_FOLDER"]
        else:
            raise ValueError(".env file does not contain DATA_DOCU_FOLDER.")
        datadoc = ddu.DataDocumentation(data_docu_folder)
        datadoc._load_data_doc()
        if "SERVER_SYMBOL" in env_dict.keys():
            datadoc.set_data_drive_symbol(env_dict["SERVER_SYMBOL"])
        session_files = datadoc.get_session_files_for_uuid(uuid)
        folder = session_files["folder"].iloc[0]
        nd2_fpath = session_files["nd2"].iloc[0]
        if isinstance(nd2_fpath, str):
            nd2_fpath = os.path.join(folder, nd2_fpath)
        else:
            print(
                f"{uuid}: nd2 file path is {nd2_fpath}, type of {type(nd2_fpath)}. Assuming no Nikon file available..."
            )
            nd2_fpath = None
        nd2_timestamps_fpath = session_files["nikon_meta"].iloc[0]
        if isinstance(nd2_timestamps_fpath, str):
            nd2_timestamps_fpath = os.path.join(folder, nd2_timestamps_fpath)
        else:
            print(f"{uuid}: No nikon timestamps file found. Assuming none available...")
            nd2_timestamps_fpath = None
        labview_fpath = session_files["labview"].iloc[0]
        if isinstance(labview_fpath, str):
            labview_fpath = os.path.join(folder, labview_fpath)
        labview_timestamps_fpath = os.path.splitext(labview_fpath)[0] + "time.txt"
        if isinstance(labview_timestamps_fpath, str):
            labview_timestamps_fpath = os.path.join(folder, labview_timestamps_fpath)
        lfp_fpath = session_files["lfp"].iloc[0]
        if isinstance(lfp_fpath, str):
            lfp_fpath = os.path.join(folder, lfp_fpath)
        else:
            lfp_fpath = None
        if matlab_2p_folder is None:
            matlab_2p_folder = env_dict["matlab_2p_folder"]
        return cls.init_and_process(
            nd2_path=nd2_fpath,
            nd2_timestamps_path=nd2_timestamps_fpath,
            labview_path=labview_fpath,
            labview_timestamps_path=labview_timestamps_fpath,
            lfp_path=lfp_fpath,
            matlab_2p_folder=matlab_2p_folder,
        )
        # except Exception:
        #    print("Setting up datadoc_util failed.")
        #    return None

    @classmethod
    def init_and_process(
        cls,
        nd2_path: str = None,
        nd2_timestamps_path: str = None,
        labview_path: str = None,
        labview_timestamps_path: str = None,
        lfp_path: str = None,
        matlab_2p_folder: str = None,
        uuid: str = None,
        **kwargs,
    ):
        """
        Instantiate a TwoPhotonSession object and perform the processing steps automatically.
        :param nd2_path: complete path of nd2 file
        :param nd2_timestamps_path: complete path of nd2 time stamps (txt) file
        :param labview_path: complete path of labview txt file (e.g. M278.20221028.133021.txt)
        :param labview_timestamps_path: complete path of labview time stamps (e.g. M278.20221028.133021time.txt)
        :param lfp_path: complete path of lfp (abf) file
        :param matlab_2p_folder: folder of matlab scripts (e.g. C:/matlab-2p/)
        :param uuid: uuid from data documentation (if exists)

        :return: The two photon session instance.
        """
        # infer rest of class attributes automatically.
        instance = cls(
            nd2_path=nd2_path,
            nd2_timestamps_path=nd2_timestamps_path,
            labview_path=labview_path,
            labview_timestamps_path=labview_timestamps_path,
            lfp_path=lfp_path,
            matlab_2p_folder=None,
            uuid=uuid,
            **kwargs,
        )
        instance._load_preprocess_data()
        if "time_total_ms" not in instance.belt_dict.keys():
            instance.belt_dict["time_total_ms"] = instance.belt_dict["time_total_s"] * 1000
        if "time_total_ms" not in instance.belt_scn_dict.keys():
            instance.belt_scn_dict["time_total_ms"] = instance.belt_scn_dict["time_total_s"] * 1000
        # convert matlab arrays into numpy arrays
        if instance.belt_dict is not None:
            for k, v in instance.belt_dict.items():
                if instance.belt_dict[k] is None:
                    instance.belt_dict[k] = np.array([])
                else:
                    instance.belt_dict[k] = instance._matlab_array_to_numpy_array(
                        instance.belt_dict[k]
                    )
            
            instance.belt_dict["dt"] = instance._lv_dt(instance.belt_dict["time_total_ms"])
            # instance.belt_dict["dt_tsscn"] = instance._lv_dt(instance.belt_dict["tsscn"])
            instance.belt_dict["totdist_abs"] = instance._lv_totdist_abs(
                instance.belt_dict["speed"], instance.belt_dict["dt"]
            )
        if instance.belt_scn_dict is not None:
            for k, v in instance.belt_scn_dict.items():
                instance.belt_scn_dict[k] = instance._matlab_array_to_numpy_array(
                    instance.belt_scn_dict[k]
                )
            instance.belt_scn_dict["dt"] = instance._lv_dt(
                instance.belt_scn_dict["time_total_ms"]
            )
            instance.belt_scn_dict["totdist_abs"] = instance._lv_totdist_abs(
                instance.belt_scn_dict["speed"], instance.belt_scn_dict["dt"]
            )
        if instance.nikon_meta is not None:
            instance._nikon_remove_na()
        instance._create_nikon_daq_time()  # defines self.nikon_daq_time
        instance._match_lfp_nikon_stamps()  # creates time_offs_lfp_nik
        if instance.nikon_daq_time is not None:
            instance._create_lfp_df(
                time_offs_lfp_nik=instance.time_offs_lfp_nik,
                cut_begin=0.0,
                cut_end=instance.nikon_daq_time.iloc[-1],
            )
        instance._create_belt_df()
        instance._create_belt_scn_df()
        if instance.nd2_path is not None:
            instance.mean_fluo = instance.return_nikon_mean()
        return instance

    def _read_out_hdf5(self, hf: h5py.File):
        # assume file has already been checked for structure, see _check_tps_hdf5_structure() above
        # get inferred entries
        dict_inferred = {}
        dict_hdf5 = {}
        for it in hf["inferred"].items():
            # these are groups inside inferred group
            if it[0] in ["belt_dict", "belt_params", "belt_scn_dict"]:
                dict_subgroup = dict()
                for it2 in it[1].items():
                    dict_subgroup[it2[0]] = it2[1][()]
                dict_inferred[it[0]] = dict_subgroup
            else:  # simple dataset
                dict_inferred[it[0]] = it[1][()]
        dict_hdf5["inferred"] = dict_inferred
        # get mean_fluo
        dict_hdf5["mean_fluo"] = hf["mean_fluo"][()]
        return dict_hdf5

    @classmethod
    def from_hdf5(cls, fpath: str, try_open_files: bool = True):
        """Open two-photon session from hdf5 file.

        Args:
            fpath (str): Path to hdf5 file
            try_open_files (bool, optional): Whether try to open the original files.
            It is the necessary for them to be accessible. Defaults to True.

        Raises:
            ValueError: If the hdf5 file does not have the proper structure.

        Returns:
            TwoPhotonSession: the opened TwoPhotonSession
        """
        # TODO: make it work with new structure of export_hdf5, incl. omitting dataframes for
        #  saving (these should be
        #  easy to recreate) if the proper flag was not set.
        # TODO: handle exceptions (missing data)
        # TODO: session.nikon_meta is not loaded
        with h5py.File(fpath, "r") as hfile:
            basic_attributes = {}
            for key, value in hfile["basic"].items():
                v = value[()]
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                basic_attributes[key.lower()] = (
                    v  # old version TwoPhotonSession hdf5 files may
                )
                # contain all capital "basic" keys
            instance = cls(
                nd2_path=basic_attributes["nd2_path"],
                nd2_timestamps_path=basic_attributes["nd2_timestamps_path"],
                labview_path=basic_attributes["labview_path"],
                labview_timestamps_path=basic_attributes["labview_timestamps_path"],
                lfp_path=basic_attributes["lfp_path"],
                matlab_2p_folder=basic_attributes["matlab_2p_folder"],
            )
            if "uuid" in hfile.attrs:
                instance.uuid = hfile.attrs["uuid"]
            # assign dictionary-type attributes
            instance.belt_dict = dict()
            instance.belt_scn_dict = dict()
            instance.belt_params = dict()
            if "inferred" not in hfile.keys():
                raise ValueError("key 'inferred' in hdf5 file not found.")
            # assume here that all three are saved (or not) together
            if "belt_dict" in hfile["inferred"].keys():
                for key, value in hfile["inferred"]["belt_dict"].items():
                    if key in constants.DICT_MATLAB_PYTHON_VARIABLES:
                        instance.belt_dict[constants.DICT_MATLAB_PYTHON_VARIABLES[key]] = value[()]
                    else:
                        instance.belt_dict[key] = value[()]
            if "belt_scn_dict" in hfile["inferred"].keys():
                for key, value in hfile["inferred"]["belt_scn_dict"].items():
                    if key in constants.DICT_MATLAB_PYTHON_SCN_VARIABLES:
                        instance.belt_scn_dict[constants.DICT_MATLAB_PYTHON_SCN_VARIABLES[key]] = value[()]
                    else:
                        instance.belt_scn_dict[key] = value[()]
            if "belt_params" in hfile["inferred"].keys():
                for key, value in hfile["inferred"]["belt_params"].items():
                    v = value[()]
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                    instance.belt_params[key] = v
            # create dict for dataframes
            if "nikon_meta" in hfile["inferred"].keys():
                nikon_meta_dict = {}
                for key, value in hfile["inferred"]["nikon_meta"].items():
                    nikon_meta_dict[key] = value[()]
                instance.nikon_meta = pd.DataFrame.from_dict(nikon_meta_dict)
            if "belt_df" in hfile["inferred"].keys():
                belt_df_dict = {}
                for key, value in hfile["inferred"]["belt_df"].items():
                    belt_df_dict[key] = value[()]
                instance.belt_df = pd.DataFrame.from_dict(belt_df_dict)
            if "belt_scn_df" in hfile["inferred"].keys():
                belt_scn_df_dict = {}
                for key, value in hfile["inferred"]["belt_scn_df"].items():
                    belt_scn_df_dict[key] = value[()]
                instance.belt_scn_df = pd.DataFrame.from_dict(belt_scn_df_dict)
            # assume lfp_df and lfp_df_cut created and saved together
            if "lfp_df" in hfile["inferred"].keys():
                lfp_df_dict = {}
                for key, value in hfile["inferred"]["lfp_df"].items():
                    lfp_df_dict[key] = value[()]
                instance.lfp_df = pd.DataFrame.from_dict(lfp_df_dict)
            if "lfp_df_cut" in hfile["inferred"].keys():
                lfp_df_cut_dict = {}
                for key, value in hfile["inferred"]["lfp_df_cut"].items():
                    lfp_df_cut_dict[key] = value[()]
                instance.lfp_df_cut = pd.DataFrame.from_dict(lfp_df_cut_dict)
            if "time_offs_lfp_nik" in hfile["inferred"].keys():
                instance.time_offs_lfp_nik = hfile["inferred"]["time_offs_lfp_nik"][()]
            if "nik_t_start" in hfile["inferred"].keys():
                n_t_s = hfile["inferred"]["nik_t_start"][()]
                if isinstance(n_t_s, bytes):
                    n_t_s = n_t_s.decode("utf-8")
                instance.nik_t_start = datetime.datetime.strptime(
                    n_t_s, DATETIME_FORMAT
                )
            if "lfp_t_start" in hfile["inferred"].keys():
                l_t_s = hfile["inferred"]["lfp_t_start"][()]
                if isinstance(l_t_s, bytes):
                    l_t_s = l_t_s.decode("utf-8")
                instance.lfp_t_start = datetime.datetime.strptime(
                    l_t_s, DATETIME_FORMAT
                )
            if "lfp_scaling" in hfile["inferred"].keys():
                instance.lfp_scaling = hfile["inferred"]["lfp_scaling"][()]
            if "nikon_daq_time" in hfile["inferred"].keys():
                instance.nikon_daq_time = pd.Series(
                    hfile["inferred"]["nikon_daq_time"][()]
                )
            if "mean_fluo" in hfile.keys():
                instance.mean_fluo = hfile["mean_fluo"][()]
                instance.nikon_true_length = len(instance.mean_fluo)
            if instance.nd2_timestamps_path is not None:
                if isinstance(instance.nd2_timestamps_path, bytes):
                    instance.nd2_timestamps_path = instance.nd2_timestamps_path.decode()

        if (
            try_open_files
        ):  # TODO: could be perfect duplicate of _open_data(). At least part of the code is duplicate
            if instance.nd2_timestamps_path is not None:
                instance._load_nikon_meta()
            if os.path.exists(instance.nd2_path):
                if isinstance(instance.nd2_path, bytes):
                    instance.nd2_path = instance.nd2_path.decode()
                instance.nikon_movie = pims_nd2.ND2_Reader(str(instance.nd2_path))
                instance.nikon_true_length = instance._find_nd2_true_length()
            else:
                print(
                    f"from_hdf5: nd2 file not found:\n\t{instance.nd2_path}. Skipping opening."
                )
            if os.path.exists(instance.lfp_path):
                if isinstance(instance.lfp_path, bytes):
                    instance.lfp_path = instance.lfp_path.decode()
                instance.lfp_file = abf.ABF(instance.lfp_path)
            else:
                print(
                    f"from_hdf5: abf file not found:\n\t{instance.lfp_path}. Skipping opening."
                )
        return instance

    def load_raw_labview_data(self):
        pass

    def _load_nikon_meta(self):
        # TODO: drop the frames where "Stimulation" is in Events Type column! (happens for jedi (high frequency) recordings)
        try:
            self.nikon_meta = self.drop_nan_cols(
                pd.read_csv(
                    self.nd2_timestamps_path, delimiter="\t", encoding="utf_16_le"
                )
            )
        except UnicodeDecodeError:
            print(
                "_open_data(): Timestamp file seems to be unusual. Trying to correct it."
            )
            output_file_path = (
                os.path.splitext(self.nd2_timestamps_path)[0] + "_corrected.txt"
            )
            ntsr.standardize_stamp_file(
                self.nd2_timestamps_path, output_file_path, export_encoding="utf_16_le"
            )
            self.nikon_meta = self.drop_nan_cols(
                pd.read_csv(output_file_path, delimiter="\t", encoding="utf_16_le")
            )
            self.nd2_timestamps_path = output_file_path
        # correct various formatting that might occur in the file
        if "Time [m:s.ms]" in self.nikon_meta.columns:
            # Example entries:
            # "0:00.2480" -> 0*60 +  0.2480
            # "3:53.1423" -> 3*60 + 53.1423
            # "2:03.9290" -> 2*60 +  3.9290

            # split by the ":" separator; convert both sides to float, multiply first element by 60, second by 1; then create sum
            print("Correcting 'Time [m:s.ms]' column...")
            self.nikon_meta["Time [m:s.ms]"] = self.nikon_meta.apply(
                lambda row: sum(
                    np.array(list(map(float, row["Time [m:s.ms]"].split(":"))))
                    * np.array([60, 1])
                ),
                axis=1,
            )
            print("Corrected.")
        if "SW Time [s]" in self.nikon_meta.columns and self.nikon_meta[
            "SW Time [s]"
        ].dtype is np.dtype("O"):
            # in this case, comma is used as separator instead of decimal point.
            print("Correcting SW Time [s] comma decimal separator...")
            self.nikon_meta["SW Time [s]"] = self.nikon_meta.apply(
                lambda row: float(row["SW Time [s]"].replace(",", ".")), axis=1
            )
            print("Corrected.")
        if "NIDAQ Time [s]" in self.nikon_meta.columns and self.nikon_meta[
            "NIDAQ Time [s]"
        ].dtype is np.dtype("O"):
            # Same as for SW Time [s]: comma is used as separator instead of decimal point.
            print("Correcting NIDAQ Time [s] comma decimal separator...")
            self.nikon_meta["NIDAQ Time [s]"] = self.nikon_meta.apply(
                lambda row: float(row["NIDAQ Time [s]"].replace(",", ".")), axis=1
            )
            print("Corrected.")

    def load_raw_data(self):
        if self.nd2_path is not None:
            self.nikon_movie = pims_nd2.ND2_Reader(self.nd2_path)
            self.nikon_true_length = self._find_nd2_true_length()
            # TODO: nikon_movie should be closed properly upon removing this class (or does the reference counter
            #  take care of it?)
        if self.nd2_timestamps_path is not None:
            self._load_nikon_meta()

    def _load_preprocess_data(self):
        """
        Load the data: Nikon 2p recording, LabView (also preprocess it), and LFP.
        :return:
        """
        if self.nd2_path is not None:
            self.nikon_movie = pims_nd2.ND2_Reader(self.nd2_path)
            self.nikon_true_length = self._find_nd2_true_length()
            # TODO: nikon_movie should be closed properly upon removing this class (or does the reference counter
            #  take care of it?)
        if self.nd2_timestamps_path is not None:
            self._load_nikon_meta()
        if (
            hasattr(self, "labview_path")
            and self.labview_path is not None
            and hasattr(self, "nd2_timestamps_path")
            and self.nd2_timestamps_path is not None
        ):
            nd2_time_stamps = ND2TimeStamps(self.nd2_timestamps_path)
            labview_time_stamps = LabViewTimeStamps(self.labview_timestamps_path)
            lv_data = LabViewData(self.labview_path)
            linear_locomotion = LinearLocomotion(
                nd2_time_stamps,
                labview_time_stamps,
                lv_data,
            )
            self.belt_df = linear_locomotion.lv_data
            self.belt_scn_df = linear_locomotion.lv_data_downsampled
            self.belt_dict = self._df_to_dict(self.belt_df)
            self.belt_scn_dict = self._df_to_dict(self.belt_scn_df)
            self.belt_params = linear_locomotion.params

        else:
            print(
                "No matching of Nikon and Labview takes place. Reason: one of the sources is missing."
            )
        if hasattr(self, "lfp_path") and self.lfp_path is not None:
            self.lfp_file = abf.ABF(self.lfp_path)
            self.lfp_scaling = LFP_SCALING_FACTOR

    @staticmethod
    def _df_to_dict(df: pd.DataFrame):
        """Given a dataframe, create a colname: df[colname] str: np.array mapping.

        Args:
            df (pd.DataFrame): _description_

        Returns:
            dict(str : np.array): _description_
        """
        return {col: df[col].values for col in df.columns}

    @staticmethod
    def _drop_useless_dimensions(array):
        """
        This function solves the problem that seems to stem from different Matlab versions used in belt processing.
        Depending on matlab version, the returned 1d array might turn into 2d: the shape of the array (x,) becomes (1,x). In terms of array elements,
        [x0, x1, ...] becomes [[x0, x1, ...]].
        This function detects if such a bad formatting occurred and attempts to correct it
        :param array: input array of matlab origin
        :return: the same data but with redundant dimension removed.
        """
        if len(array.shape) == 1:
            return array
        if len(array.shape) == 2 and array.shape[0] == 1:
            # print(f"Matlab possibly messed up an array, shape {array.shape} detected; should probably be ({array.shape[1]},). attempting to remove the redundant dimension...")
            return array[0]
        return array

    def drop_nan_cols(self, dataframe: pd.DataFrame):
        to_drop = []
        for column in dataframe.columns:
            if len(dataframe[column].dropna()) == 0:
                to_drop.append(column)
        return dataframe.drop(to_drop, axis="columns")

    def _matlab_array_to_numpy_array(self, matlab_array):
        if type(matlab_array) is np.ndarray:
            return self._drop_useless_dimensions(matlab_array)
        else:
            return self._drop_useless_dimensions(np.array(matlab_array._data))

    def _nikon_remove_na(self):
        # get the stimulation metadata frames
        if self.nikon_meta is None:
            warnings.warn("_nikon_remove_na(): nikon metadata not available.")
        else:
            df_stim = self.nikon_meta[self.nikon_meta["Index"].isna() == True]
            # FIXME low priority -  Turns out there are so many issues with the recorded time
            self.df_stim = df_stim
            # stamps (often false entries) that it's not worth it now to use them.
            # Use data documentation stimulation frames instead.

            # drop non-imaging frames from metadata
            self.nikon_meta.dropna(subset=["Index"], inplace=True)

    def _lfp_movement_raw(self):
        self.lfp_file.setSweep(sweepNumber=0, channel=1)
        return self.lfp_file.sweepX, self.lfp_file.sweepY

    def _lfp_lfp_raw(self):
        self.lfp_file.setSweep(sweepNumber=0, channel=0)
        return self.lfp_file.sweepX, self.lfp_file.sweepY

    def lfp_movement(self, as_numpy: bool = False):
        """
        Returns columns of the lfp data internal dataframe as pandas Series (or numpy array if as_numpy is True): a tuple of (t_series, y_series) for lfp
        channel 0 (lfp) data.
        :return: tuple of two pandas Series
        """
        if as_numpy:
            return (
                self.lfp_df["t_mov_corrected"].to_numpy(),
                self.lfp_df["y_mov"].to_numpy(),
            )
        else:
            return self.lfp_df["t_mov_corrected"], self.lfp_df["y_mov"]

    def lfp_lfp(self, as_numpy: bool = False):
        """
        Returns columns of the lfp data internal dataframe as pandas Series (or numpy array if as_numpy is True): a tuple of (t_series, y_series) for lfp
        channel 1 (movement) data.
        :return: tuple of two pandas Series
        """
        if as_numpy:
            return (
                self.lfp_df["t_lfp_corrected"].to_numpy(),
                self.lfp_df["y_lfp"].to_numpy(),
            )
        else:
            return self.lfp_df["t_lfp_corrected"], self.lfp_df["y_lfp"]

    def labview_movement(self, as_numpy: bool = False):
        """
        Returns columns of the labview data internal dataframe as pandas Series (or numpy array if as_numpy is True): a tuple of (t_series, y_series)
        for labView time and speed data.
        :return: tuple of two pandas Series
        """
        if as_numpy:
            return self.belt_df.time_s.to_numpy(), self.belt_df.speed.to_numpy()
        else:
            return self.belt_df.time_s, self.belt_df.speed

    def _create_nikon_daq_time(self):
        if self.nikon_meta is None:
            warnings.warn("_create_nikon_daq_time: nikon metadata is not available.")
        else:
            self.nikon_daq_time = self.nikon_meta["NIDAQ Time [s]"]
            # no change needed
            if isinstance(self.nikon_daq_time.iloc[0], float):
                pass
            elif isinstance(self.nikon_daq_time.iloc[0], str):
                # if elements are string, they seem to have comma as decimal separator. Need to replace it by a dot.
                self.nikon_daq_time = self.nikon_daq_time.apply(
                    lambda s: float(s.replace(",", "."))
                )
            else:  # something went really wrong!
                raise ValueError(
                    f"nikon_daq_time has unsupported data type: {type(self.nikon_daq_time.iloc[0])}"
                )

    def _lv_totdist_abs(self, speed, dt) -> np.array:
        """
        Create a true total distance statistic. totdist from labview integrates the distance with sign, i.e. the total
        distance is reduced with backwards locomotion.
        :return: numpy array
        """
        assert len(speed) == len(dt)
        totdist_abs = np.zeros(len(speed))
        totdist_abs[0] = speed[0] * dt[0]
        for i in range(1, len(totdist_abs)):
            totdist_abs[i] = totdist_abs[i - 1] + abs(speed[i] * dt[i])
        return totdist_abs

    def _lv_dt(self, t) -> np.array:
        """
        Create new array with entry i = t[i] - t[i-1], with dt[0] = 0
        :param t: a time series. 1d numpy array
        :return: numpy array with dt entries, same length as t
        """
        print(f"Shape of t in _lv_dt is {t.shape}")
        t1 = t[1:]
        t0 = t[:-1]
        dt = np.zeros(len(t))
        dt[1:] = t1 - t0
        dt[0] = dt[1]  # assume same time step to avoid having a 0.
        return dt

    def shift_lfp(self, seconds: float = 0.0, match_type: str = "Nikon") -> None:
        """
        Shifts the LFP signal (+/-) by the amount of seconds: an event at time t to time (t+seconds), i.e. a positive
        seconds means shifting everything to a later time.
        match_type: "Nikon" or "zero". "Nikon": match (cut) to the Nikon frames (NIDAQ time stamps). "zero": match to
        0 s, and the last Nikon frame.
        """
        cut_end = self.nikon_daq_time.iloc[-1]
        if match_type == "Nikon":  # TODO: match_type does not explain its own function
            cut_begin = self.nikon_daq_time.iloc[0]
        else:
            cut_begin = 0.0
        self._create_lfp_df(self.time_offs_lfp_nik - seconds, cut_begin, cut_end)
        self.time_offs_lfp_nik = self.time_offs_lfp_nik - seconds

    def nikon_time_stamp(self, i_frame):
        """
        Given the 0-indexed i_frame of the Nikon recording, return the time stamp associated with it in UTC. Raises
        Exception if i_frame is not in the valid range.
        :param i_frame: 0-indexed frame in range [0, <length of Nikon movie>]
        :return: datetime object of time stamp for frame.
        """
        tzone_utc = pytz.utc
        if (i_frame < len(self.nikon_movie)) and (i_frame >= 0):
            t_start = tzone_utc.localize(self.nikon_movie.metadata["time_start_utc"])
            dt_ms = datetime.timedelta(
                milliseconds=self.nikon_movie[i_frame].metadata["t_ms"]
            )
            return t_start + dt_ms
        else:
            raise Exception(
                f"TwoPhotonSession nikon_time_stamp(i_frame): i_frame ({i_frame}) out of range [0, "
                f"{len(self.nikon_movie) - 1}]"
            )
        pass

    # TODO: this does not actually matches the two, but gets the offset for matching
    def _match_lfp_nikon_stamps(self) -> None:
        if (
            hasattr(self, "lfp_file")
            and self.lfp_file is not None
            and self.nikon_movie is not None
        ):
            # time zone of the recording computer
            tzone_local = pytz.timezone("Europe/Berlin")
            tzone_utc = pytz.utc

            lfp_t_start: datetime.datetime = tzone_local.localize(
                self.lfp_file.abfDateTime
            )  # supply timezone information
            try:
                nik_t_start: datetime.datetime = tzone_utc.localize(
                    self.nikon_movie.metadata["time_start_utc"]
                )
            except (
                Exception
            ):  # in case of exception, a corruption might have happened, so last part of
                # metadata is missing.
                # code of @property metadata() from nd2reader.py (pims_nd2)
                warnings.warn(
                    "Error reading out metadata of nd2. Recording might be corrupted; most likely\
                     later time stamps. Check for repeated values in exported _nik.txt meta file. \
                     Attempting to read out only first time stamp..."
                )
                nik_t_start: datetime.datetime = tzone_utc.localize(
                    pims_nd2.ND2SDK.jdn_to_datetime_utc(
                        self.nikon_movie._lim_metadata_desc.dTimeStart
                    )
                )

            # now both can be converted to utc
            lfp_t_start = lfp_t_start.astimezone(pytz.utc)
            nik_t_start = nik_t_start.astimezone(pytz.utc)

            # save these as instance variables
            self.lfp_t_start = lfp_t_start
            self.nik_t_start = nik_t_start

            # FIXME: correcting for time offset is not enough! The nikon DAQ time also does not start from 0!

            # avoid deltatime issue: when earlier - later, the delta time will be negative days, huge number of hours.
            # For example: -1 days, 23:59:59 hours.
            if nik_t_start > lfp_t_start:
                sign = 1.0
            else:
                sign = -1.0

            time_offs_lfp_nik = abs(nik_t_start - lfp_t_start)
            time_offset_sec = sign * time_offs_lfp_nik.seconds

            time_offs_lfp_nik = time_offset_sec + (
                sign * time_offs_lfp_nik.microseconds * 1e-6
            )

            # stop process if too much time detected between starting LFP and Nikon recording.
            if abs(time_offs_lfp_nik) > 30.0:
                warnings.warn(
                    f"Warning! more than 30 s difference detected between starting the LFP and the Nikon "
                    f"recording!\nPossible cause: bug in conversion to utc (daylight saving mode, "
                    f"timezone conversion).\nlfp: {lfp_t_start}\nnikon: {nik_t_start}\noffset (s): {time_offs_lfp_nik}"
                )

            print(f"Difference of starting times (s): {time_offs_lfp_nik}")
        else:
            time_offs_lfp_nik = None
        self.time_offs_lfp_nik = time_offs_lfp_nik

    def _find_nd2_true_length(self) -> int:
        """
        The default value, len(self.nikon_movie), in case of corruption, is greater than the actual accessible length.
        This function checks the last frame that is accessible, and returns the total length from frame 1 to the last
        accessible frame.
        :return: int, the true length. If the last frame is readable, this is equal to len(self.nikon_movie)
        """
        if self.nikon_movie is not None:
            i = len(self.nikon_movie) - 1
            frame_read_success = False
            while not frame_read_success:
                try:
                    fr = self.nikon_movie[i]
                    frame_read_success = True
                    # TODO: could just return the detected length here. Not sure about asynchronous events (is
                    #  Exception caught immediately?)
                except Exception:  # TODO: separate KeyboardInterrupt!
                    i -= 1
                    frame_read_success = False
                if i < 0:
                    raise ValueError("self.nikon_movie cannot be read!")
            return i + 1
        else:
            warnings.warn(
                "Warning: _find_nd2_true_length() called, but self.nikon_movie is None! Returning 0."
            )
            return 0

    def _belt_dict_to_df(self, belt_dict: dict) -> pd.DataFrame:
        """
        This function takes belt_dict or belt_scn_dict and returns it as a dataframe. Some columns with less entries
        are removed!
        """
        if belt_dict is None:
            return None
        # only reliable way I know of to differentiate between belt and belt_scn.
        if "runtime" in belt_dict:
            bd = belt_dict.copy()
            bd.pop("runtime")
            if "tsscn" in bd:
                bd.pop("tsscn")
            df = pd.DataFrame(bd)
        else:
            df = pd.DataFrame(belt_dict)
        if "time" in df.columns:
            df["time_s"] = df["time"] / 1000.0
        return df

    def _create_belt_df(self):
        self.belt_df = self._belt_dict_to_df(self.belt_dict)

    def _create_belt_scn_df(self):
        self.belt_scn_df = self._belt_dict_to_df(self.belt_scn_dict)

    def _create_lfp_cut_df(
        self, lfp_df_raw, lower_limit: float, upper_limit: float
    ) -> pd.DataFrame:
        lfp_df_new_cut = lfp_df_raw[lfp_df_raw["t_lfp_corrected"] >= lower_limit]
        lfp_df_new_cut = lfp_df_new_cut[
            lfp_df_new_cut["t_lfp_corrected"] <= upper_limit
        ]
        return lfp_df_new_cut

    def _create_lfp_df(
        self, time_offs_lfp_nik: float, cut_begin: float, cut_end: float
    ):
        if hasattr(self, "lfp_file") and self.lfp_file is not None:
            lfp_df_new = pd.DataFrame()
            lfp_df_new_cut = pd.DataFrame()
            t_mov, y_mov = self._lfp_movement_raw()
            t_lfp, y_lfp = self._lfp_lfp_raw()
            # add movement data
            lfp_df_new["t_mov_raw"] = t_mov
            lfp_df_new["t_mov_offset"] = lfp_df_new["t_mov_raw"] - time_offs_lfp_nik
            # scale factor given in Bence's excel sheets
            lfp_df_new["t_mov_corrected"] = (
                lfp_df_new["t_mov_offset"] * LFP_SCALING_FACTOR
            )
            lfp_df_new["y_mov"] = y_mov
            # add normalized movement data
            motion_min = lfp_df_new["y_mov"].min()
            motion_max = lfp_df_new["y_mov"].max()
            motion_mean = lfp_df_new["y_mov"].mean()
            # FIXME: normalization should be (y - mean)/(max - min)
            lfp_df_new["y_mov_normalized"] = lfp_df_new["y_mov"] / motion_mean

            # add lfp data
            lfp_df_new["t_lfp_raw"] = t_lfp
            lfp_df_new["t_lfp_offset"] = lfp_df_new["t_lfp_raw"] - time_offs_lfp_nik
            # scale factor given in Bence's excel sheets
            # TODO: document columns of dataframes. corrected vs offset
            lfp_df_new["t_lfp_corrected"] = (
                lfp_df_new["t_lfp_offset"] * LFP_SCALING_FACTOR
            )
            lfp_df_new["y_lfp"] = y_lfp

            # cut lfp data
            # cut to Nikon. LFP will not start at 0!
            # lfp_df_new_cut = self._create_lfp_cut_df(
            #    lfp_df_new, self.nikon_daq_time.iloc[0], self.nikon_daq_time.iloc[-1])
            lfp_df_new_cut = self._create_lfp_cut_df(lfp_df_new, cut_begin, cut_end)
            self.lfp_df, self.lfp_df_cut = lfp_df_new, lfp_df_new_cut
        else:
            print("TwoPhotonSession: LFP file was not specified.")
            self.lfp_df, self.lfp_df_cut = None, None

        # now nd2_to_caiman.py

    def has_lfp(self):
        if self.lfp_df is not None:
            return True
        else:
            return False

    def get_nikon_data(self, i_begin: int = None, i_end: int = None) -> np.array:
        """
        :param i_begin: 0-indexed first frame to get
        :param i_end: 0-indexed last frame to get
        :return:
        """
        # TODO: test this function properly
        # TODO: i_begin OR i_end not defined, set them to 0 or last frame, respectively
        # set iter_axes to "t"
        # then: create nd array with sizes matching frame size,
        sizes_dict = self.nikon_movie.sizes
        true_len = self._find_nd2_true_length()
        if "t" in sizes_dict.keys() and sizes_dict["t"] > true_len:
            warnings.warn(
                "Warning: get_nikon_data called on corrupt file. Will not use corrupt, inaccessible frames"
            )
            sizes_dict["t"] = true_len
        pixel_type = self.nikon_movie.pixel_type
        if (i_begin is not None) and (i_end is not None):
            n_frames = i_end - i_begin
            i_first = i_begin
            assert i_end < sizes_dict["t"]
        else:
            n_frames = sizes_dict["t"]
            i_first = 0
        sizes = (n_frames, sizes_dict["x"], sizes_dict["y"])
        # dtype would be float32 by default...
        frames_arr = np.zeros(sizes, dtype=pixel_type)
        for i_frame in range(n_frames):
            frames_arr[i_frame] = np.array(
                self.nikon_movie[i_first + i_frame], dtype=pixel_type
            )  # not sure if dtype needed here
        return frames_arr

    # TODO: handle missing files
    # FIXME: if lfp missing, no inferred group is created!
    def export_hdf5(self, fpath: str = None, save_full: bool = False, **kwargs) -> str:
        """
        Parameters to export:
                creation_time: str(datetime.now()) for version checking
            Basic attributes:
                self.uuid
                self.nd2_path
                self.nd2_timestamps_path
                self.labview_path
                self.labview_timestamps_path
                self.lfp_path
                self.matlab_2p_folder
            Inferred attributes:
                self.belt_dict
                self.belt_scn_dict
                self.time_offs_lfp_nik
                self.lfp_t_start
                self.nik_t_start
                self.lfp_scaling
                self.belt_params
                self.nikon_daq_time - Series
                self.mean_fluo
                Optionally saved (save_full = True):
                    [lfp_df] - DataFrame
                    [lfp_df_cut] - DataFrame
                    [belt_df] - DataFrame
                    [belt_scn_df] - DataFrame
                    [nikon_meta] - DataFrame
            Not saved:
                (lfp_file) - ABFReader
                (nikon_movie) - ND2Reader

        :param kwargs:
        fpath: str - the currently not existing file (including folder and file name) to be created
        and saved to.
            Should have ".h5" extension. For example: "C:\\Downloads\\session_v1.h5"
        save_full: bool - flag whether to save redundant dataframes (i.e. the full object,
        except the data sources).
            Note: if the result is to be used on a computer where the original files are not
            accessible, it makes sense to set this flag to True. In this case, not only mean
            Nikon fluorescence and matched and raw labview data, but also matched LFP data will
            be saved in the file.
        :return: fpath: the exported file path as string.
        """
        # set export file name and path
        if fpath is None:
            fpath = kwargs.get("fpath", os.path.splitext(self.nd2_path)[0] + ".h5")
        with h5py.File(fpath, "w") as hfile:
            hfile.attrs["creation_time"] = str(datetime.datetime.now())
            if self.uuid is not None:
                hfile.attrs["uuid"] = self.uuid
            else:
                warnings.warn(
                    "No uuid given! Consider assigning one and exporting again."
                )
                hfile.attrs["uuid"] = "NaN"

            if self.nikon_movie is not None:
                hfile.attrs["nikon_true_length"] = self.nikon_true_length
                hfile.attrs["nikon_orig_length"] = len(self.nikon_movie)
            if self.mean_fluo is not None:
                hfile.create_dataset("mean_fluo", data=self.mean_fluo)
            basic_group = hfile.create_group("basic")
            # basic parameters
            basic_group["nd2_path"] = self.nd2_path if self.nd2_path is not None else ""
            basic_group["nd2_timestamps_path"] = (
                self.nd2_timestamps_path if self.nd2_timestamps_path is not None else ""
            )
            basic_group["labview_path"] = (
                self.labview_path if self.labview_path is not None else ""
            )
            basic_group["labview_timestamps_path"] = (
                self.labview_timestamps_path
                if self.labview_timestamps_path is not None
                else ""
            )
            basic_group["lfp_path"] = self.lfp_path if self.lfp_path is not None else ""
            basic_group["matlab_2p_folder"] = (
                self.matlab_2p_folder if self.matlab_2p_folder is not None else ""
            )
            # implied parameters
            inferred_group = hfile.create_group("inferred")
            # save nikon_meta as group with columns as datasets
            # nikon_meta_group = inferred_group.create_group("nikon_meta")
            # if self.nikon_meta is not None:
            #    for col_name in self.nikon_meta.keys():
            #        nikon_meta_group[col_name] = self.nikon_meta[col_name].to_numpy()
            # save belt_dict
            belt_dict_group = inferred_group.create_group("belt_dict")
            if self.belt_dict is not None:
                for key, value in self.belt_dict.items():
                    belt_dict_group[key] = value
            # save belt_scn_dict
            if self.belt_scn_dict is not None:
                belt_scn_dict_group = inferred_group.create_group("belt_scn_dict")
                for key, value in self.belt_scn_dict.items():
                    belt_scn_dict_group[key] = value
            # save pandas Series nikon_daq_time
            if self.nikon_daq_time is not None:
                inferred_group["nikon_daq_time"] = self.nikon_daq_time.to_numpy()
            # save time_offs_lfp_nik
            if self.time_offs_lfp_nik is not None:
                inferred_group["time_offs_lfp_nik"] = (
                    self.time_offs_lfp_nik
                    if self.time_offs_lfp_nik is not None
                    else np.nan
                )
            # save belt_params
            if self.belt_params is not None:
                belt_params_group = inferred_group.create_group("belt_params")
                for key, value in self.belt_params.items():
                    belt_params_group[key] = value
            # save lfp_t_start, nik_t_start, lfp_scaling if available
            inferred_group["lfp_t_start"] = (
                self.lfp_t_start.strftime(DATETIME_FORMAT)
                if self.lfp_t_start is not None
                else ""
            )
            inferred_group["nik_t_start"] = (
                self.nik_t_start.strftime(DATETIME_FORMAT)
                if self.nik_t_start is not None
                else ""
            )
            inferred_group["lfp_scaling"] = (
                self.lfp_scaling if self.lfp_scaling is not None else np.nan
            )

            # save lfp_df
            if self.lfp_df is not None and save_full:
                lfp_df_group = inferred_group.create_group("lfp_df")
                for col_name, col in self.lfp_df.items():
                    lfp_df_group[col_name] = col.to_numpy()
            # save lfp_df_cut
            if self.lfp_df_cut is not None and save_full:
                lfp_df_cut_group = inferred_group.create_group("lfp_df_cut")
                for col_name in self.lfp_df_cut.keys():
                    lfp_df_cut_group[col_name] = self.lfp_df_cut[col_name].to_numpy()
            # belt_df and belt_scn_df are duplicates of belt_dict and belt_scn_dict, no need to save them.
            # save belt_df
            # if self.belt_df is not None and save_full:
            #    belt_df_group = inferred_group.create_group("belt_df")
            #    for col_name in self.belt_df.keys():
            #        belt_df_group[col_name] = self.belt_df[col_name].to_numpy()
            # save belt_scn_df
            # if self.belt_scn_df is not None and save_full:
            #    belt_scn_df_group = inferred_group.create_group("belt_scn_df")
            #    for col_name in self.belt_scn_df.keys():
            #        belt_scn_df_group[col_name] = self.belt_scn_df[col_name].to_numpy()

            if self.nikon_meta is not None and save_full:
                nikon_meta_group = inferred_group.create_group("nikon_meta")
                for col_name in self.nikon_meta.keys():
                    if self.nikon_meta[col_name].dtype == np.dtype("O"):
                        print(
                            f"nikon_meta['{col_name}'] entries have type 'np.dtype('O')'. Converting..."
                        )
                        nikon_meta_group[col_name] = self.nikon_meta[col_name].to_numpy(
                            dtype=np.float64
                        )
                    else:
                        nikon_meta_group[col_name] = self.nikon_meta[
                            col_name
                        ].to_numpy()
        return fpath

    # TODO: get nikon frame matching time stamps (NIDAQ time)! It is session.nikon_daq_time
    def return_nikon_mean(self):
        """Calculate the mean trace of the nikon movie. Update relevant attributes,
        and return the trace.

        Returns:
            np.array: The mean trace.
        """
        if self.nikon_true_length is None:
            self.nikon_true_length = self._find_nd2_true_length()
        try:
            if self.nikon_true_length < len(self.nikon_movie):
                warnings.warn(
                    "Warning: self.nikon_true_length is smaller than length of self.nikon_movie.\
                     This means most likely corrupted frames. Part of recording can not be opened.\
                      Take this into consideration in further analysis."
                )
            arr = np.array(
                [
                    self.nikon_movie[i_frame].mean()
                    for i_frame in range(self.nikon_true_length)
                ]
            )
            return arr
        except KeyboardInterrupt:
            warnings.warn(
                "return_nikon_mean: Keyboard interrupt detected. Returning empty np.array()."
            )
            return np.array([])
        except Exception:
            warnings.warn(
                "Error reading out nd2 file; it seems to be corrupted. It might be possible to save it to tiff using "
                "the nikon software, and calculate the mean from that."
            )

    def infer_labview_timestamps(self):
        """
        Try to infer the labview time stamps filename given the labview filename.
        :return: None
        """
        if self.labview_path is not None:
            inferred_fpath = os.path.splitext(self.labview_path)[0] + "time.txt"
            if os.path.exists(inferred_fpath):
                if self.labview_timestamps_path is None:
                    self.labview_timestamps_path = inferred_fpath
                    print(
                        f"Inferred labview timestamps file path:\n\t{self.labview_timestamps_path}"
                    )
                else:  # timestamps file already defined
                    print(
                        f"Labview timestamps file seems to already exist:\n\t{self.labview_timestamps_path}\nNOT "
                        f"changing it."
                    )
        else:
            print(
                "Can not infer labview timestamps filename, as no labview data was defined. (txt file with labview "
                "readout data)"
            )


def open_session(data_path: str) -> TwoPhotonSession:
    """Given a folder path, open a TwoPhotonSession instance by selecting the necessary files.

    Args:
        data_path (str): The folder where the files might be located (initial directory for
        file dialog).

    Returns:
        TwoPhotonSession: The opened two photon session.
    """
    # TODO: test this function! Make it an alternative constructor
    # .nd2 file
    nd2_path = askopenfilename(initialdir=data_path, title="Select .nd2 file")
    print(f"Selected imaging file: {nd2_path}")

    # nd2 info file (..._nik.txt) Image Proterties -> Recorded Data of .nd2 file saved as .txt
    nd2_timestamps_path = os.path.splitext(nd2_path)[0] + "_nik.txt"
    if not os.path.exists(nd2_timestamps_path):
        nd2_timestamps_path = askopenfilename(
            initialdir=data_path, title="Nikon info file not found. Please provide it!"
        )
    print(f"Selected nd2 info file: {nd2_timestamps_path}")

    # labview .txt file
    labview_path = askopenfilename(
        initialdir=data_path, title="Select corresponding labview (xy.txt) file"
    )
    print(f"Selected LabView data file: {labview_path}")

    # labview time stamp (...time.txt)
    labview_timestamps_path = (
        os.path.splitext(labview_path)[0] + "time.txt"
    )  # try to open the standard corresponding time stamp file first
    if not os.path.exists(labview_timestamps_path):
        labview_timestamps_path = askopenfilename(
            initialdir=data_path,
            title="Labview time stamp not found. Please provide it!",
        )
    print(f"Selected LabView time stamp file: {labview_timestamps_path}")

    # lfp file (.abf)
    lfp_path = askopenfilename(initialdir=data_path, title="Select LFP .abf file")
    print(f"Selected LFP file: {lfp_path}")

    session = TwoPhotonSession(
        nd2_path=nd2_path,
        nd2_timestamps_path=nd2_timestamps_path,
        labview_path=labview_path,
        labview_timestamps_path=labview_timestamps_path,
        lfp_path=lfp_path,
    )
    return session


# TODO: extract these methods to a new python file, and move imports outside functions to speed up.
# taken from caiman.utils.visualization.py
def nb_view_patches_with_lfp_movement(
    Yr,
    A,
    C,
    b,
    f,
    d1,
    d2,
    t_lfp: np.array = None,
    y_lfp: np.array = None,
    t_mov: np.array = None,
    y_mov: np.array = None,
    YrA=None,
    image_neurons=None,
    thr=0.99,
    denoised_color=None,
    cmap="jet",
    r_values=None,
    SNR=None,
    cnn_preds=None,
):
    """
    Interactive plotting utility for ipython notebook

    Args:
        Yr: np.ndarray
            movie

        A,C,b,f: np.ndarrays
            outputs of matrix factorization algorithm

        d1,d2: floats
            dimensions of movie (x and y)

        YrA:   np.ndarray
            ROI filtered residual as it is given from update_temporal_components
            If not given, then it is computed (K x T)

        image_neurons: np.ndarray
            image to be overlaid to neurons (for instance the average)

        thr: double
            threshold regulating the extent of the displayed patches

        denoised_color: string or None
            color name (e.g. 'red') or hex color code (e.g. '#F0027F')

        cmap: string
            name of colormap (e.g. 'viridis') used to plot image_neurons
            :param r_values:
    """

    colormap = mpl.cm.get_cmap(cmap)
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape
    nA2 = (
        np.ravel(np.power(A, 2).sum(0))
        if isinstance(A, np.ndarray)
        else np.ravel(A.power(2).sum(0))
    )
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        Y_r = np.array(
            spdiags(old_div(1, nA2), 0, nr, nr)
            * (
                A.T * np.matrix(Yr)
                - (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis])
                - A.T.dot(A) * np.matrix(C)
            )
            + C
        )
    else:
        Y_r = C + YrA

    x = np.arange(T)
    if image_neurons is None:
        image_neurons = A.mean(1).reshape((d1, d2), order="F")

    coors = get_contours(A, (d1, d2), thr)
    cc1 = [cor["coordinates"][:, 0] for cor in coors]
    cc2 = [cor["coordinates"][:, 1] for cor in coors]
    c1 = cc1[0]
    c2 = cc2[0]

    # split sources up, such that Bokeh does not warn
    # "ColumnDataSource's columns must be of the same length"
    source = ColumnDataSource(data=dict(x=x, y=Y_r[0] / 100, y2=C[0] / 100))
    source_ = ColumnDataSource(data=dict(z=Y_r / 100, z2=C / 100))
    source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
    source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))

    code = """
            var data = source.data
            var data_ = source_.data
            var f = cb_obj.value - 1
            var x = data['x']
            var y = data['y']
            var y2 = data['y2']

            for (var i = 0; i < x.length; i++) {
                y[i] = data_['z'][i+f*x.length]
                y2[i] = data_['z2'][i+f*x.length]
            }

            var data2_ = source2_.data;
            var data2 = source2.data;
            var c1 = data2['c1'];
            var c2 = data2['c2'];
            var cc1 = data2_['cc1'];
            var cc2 = data2_['cc2'];

            for (var i = 0; i < c1.length; i++) {
                   c1[i] = cc1[f][i]
                   c2[i] = cc2[f][i]
            }
            source2.change.emit();
            source.change.emit();
        """

    if r_values is not None:
        code += """
            var mets = metrics.data['mets']
            mets[1] = metrics_.data['R'][f].toFixed(3)
            mets[2] = metrics_.data['SNR'][f].toFixed(3)
            metrics.change.emit();
        """
        metrics = ColumnDataSource(
            data=dict(
                y=(3, 2, 1, 0),
                mets=(
                    "",
                    "% 7.3f" % r_values[0],
                    "% 7.3f" % SNR[0],
                    (
                        "N/A"
                        if np.sum(cnn_preds) in (0, None)
                        else "% 7.3f" % cnn_preds[0]
                    ),
                ),
                keys=("Evaluation Metrics", "Spatial corr:", "SNR:", "CNN:"),
            )
        )
        if np.sum(cnn_preds) in (0, None):
            metrics_ = ColumnDataSource(data=dict(R=r_values, SNR=SNR))
        else:
            metrics_ = ColumnDataSource(data=dict(R=r_values, SNR=SNR, CNN=cnn_preds))
            code += """
                mets[3] = metrics_.data['CNN'][f].toFixed(3)
            """
        labels = LabelSet(x=0, y="y", text="keys", source=metrics, render_mode="canvas")
        labels2 = LabelSet(
            x=10,
            y="y",
            text="mets",
            source=metrics,
            render_mode="canvas",
            text_align="right",
        )
        plot2 = bpl.figure(plot_width=200, plot_height=100, toolbar_location=None)
        plot2.axis.visible = False
        plot2.grid.visible = False
        plot2.tools.visible = False
        plot2.line([0, 10], [0, 4], line_alpha=0)
        plot2.add_layout(labels)
        plot2.add_layout(labels2)
    else:
        metrics, metrics_ = None, None

    callback = CustomJS(
        args=dict(
            source=source,
            source_=source_,
            source2=source2,
            source2_=source2_,
            metrics=metrics,
            metrics_=metrics_,
        ),
        code=code,
    )

    plot = bpl.figure(plot_width=600, plot_height=200, x_range=Range1d(0, Y_r.shape[0]))
    plot.line("x", "y", source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line(
            "x", "y2", source=source, line_width=1, line_alpha=0.6, color=denoised_color
        )

    xr = Range1d(start=0, end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0], end=0)
    plot1 = bpl.figure(
        x_range=xr,
        y_range=yr,
        plot_width=int(min(1, d2 / d1) * 300),
        plot_height=int(min(1, d1 / d2) * 300),
    )

    plot1.image(
        image=[image_neurons[::-1, :]],
        x=0,
        y=image_neurons.shape[0],
        dw=d2,
        dh=d1,
        palette=grayp,
    )
    plot1.patch("c1", "c2", alpha=0.6, color="purple", line_width=2, source=source2)

    # create plot for lfp
    if y_lfp is not None:
        source_lfp = ColumnDataSource(data=dict(x=t_lfp, y=y_lfp))
        plot_lfp = bpl.figure(
            x_range=Range1d(t_lfp[0], t_lfp[-1]),
            y_range=Range1d(y_lfp.min(), y_lfp.max()),
            plot_width=plot.plot_width,
            plot_height=plot.plot_height,
        )
        plot_lfp.line("x", "y", source=source_lfp)
    # plot_mov = bpl.figure(x_range=xr, y_range=None)
    if y_mov is not None:
        source_mov = ColumnDataSource(data=dict(x=t_mov, y=y_mov))
        plot_mov = bpl.Figure(
            x_range=Range1d(t_mov[0], t_mov[-1]),
            y_range=Range1d(y_mov.min(), y_mov.max()),
            plot_width=plot.plot_width,
            plot_height=plot.plot_height,
        )
        plot_mov.line("x", "y", source=source_mov)
    if Y_r.shape[0] > 1:
        slider = Slider(
            start=1, end=Y_r.shape[0], value=1, step=1, title="Neuron Number"
        )
        slider.js_on_change("value", callback)
        if y_mov is not None:
            if y_lfp is not None:  # both lfp and mov
                bpl.show(
                    layout(
                        [
                            [slider],
                            [
                                row(
                                    plot1 if r_values is None else column(plot1, plot2),
                                    column(plot, plot_lfp, plot_mov),
                                )
                            ],
                        ]
                    )
                )
            else:  # no lfp plot
                bpl.show(
                    layout(
                        [
                            [slider],
                            [
                                row(
                                    plot1 if r_values is None else column(plot1, plot2),
                                    column(plot, plot_mov),
                                )
                            ],
                        ]
                    )
                )
        else:  # no mov plot
            if y_lfp is not None:
                bpl.show(
                    layout(
                        [
                            [slider],
                            [
                                row(
                                    plot1 if r_values is None else column(plot1, plot2),
                                    column(plot, plot_lfp),
                                )
                            ],
                        ]
                    )
                )
            else:  # no lfp and no movement
                bpl.show(
                    layout(
                        [
                            [slider],
                            [
                                row(
                                    plot1 if r_values is None else column(plot1, plot2),
                                    plot,
                                )
                            ],
                        ]
                    )
                )
    else:
        bpl.show(row(plot1 if r_values is None else column(plot1, plot2), plot))

    return Y_r


# TODO: extract these methods to a new python file, and move imports outside functions to speed up.
# taken from caiman.utils.visualization.py
def nb_view_patches_manual_control_NOTWORKING(
    Yr,
    A,
    C,
    b,
    f,
    d1,
    d2,
    YrA=None,
    image_neurons=None,
    thr=0.99,
    denoised_color=None,
    cmap="jet",
    r_values=None,
    SNR=None,
    cnn_preds=None,
    mode: str = None,
    idx_accepted: List = None,
    idx_rejected: List = None,
):
    """
    Interactive plotting utility for ipython notebook
    Sadly, does not work, probably because of overflow.
    Args:
        Yr: np.ndarray
            movie

        A,C,b,f: np.ndarrays
            outputs of matrix factorization algorithm

        d1,d2: floats
            dimensions of movie (x and y)

        YrA:   np.ndarray
            ROI filtered residual as it is given from update_temporal_components
            If not given, then it is computed (K x T)

        image_neurons: np.ndarray
            image to be overlaid to neurons (for instance the average)

        thr: double
            threshold regulating the extent of the displayed patches

        denoised_color: string or None
            color name (e.g. 'red') or hex color code (e.g. '#F0027F')

        cmap: string
            name of colormap (e.g. 'viridis') used to plot image_neurons
            :param r_values:

        mode: string
            The initial view mode to show. It should be one of 'accepted', 'rejected', 'all'. The fourth option,
            'modified', would cause an empty plot.

        idx_accepted: List
            The idx_components field of the estimates object.

        idx_rejected: List
            The idx_components_bad field of the estimates object.
    """

    # TODO: idx_components and idx_components_bad refer to indices of accepted/rejected neurons, use these in
    #  nb_view_components_manual_control. If These don't exist, that means select_components has been called... I don't
    #  know if it is still possible (easily) to move the neurons from one group to the other.
    #    nb_view_patches_manual_control(
    #    Yr, estimates.A.tocsc()[:, idx], estimates.C[idx], estimates.b, estimates.f,
    #    estimates.dims[0], estimates.dims[1],
    #    YrA=estimates.R[idx], image_neurons=img,
    #    thr=thr, denoised_color=denoised_color, cmap=cmap,
    #    r_values=None if estimates.r_values is None else estimates.r_values[idx],
    #    SNR=None if estimates.SNR_comp is None else estimates.SNR_comp[idx],
    #    cnn_preds=None if np.sum(estimates.cnn_preds) in (0, None) else estimates.cnn_preds[idx],
    #    mode=mode)
    # No easy way to use these in CustomJS. Could define beginning of variable 'code' like this, and append the rest
    # REJECTED_COLOR = "red"
    # REJECTED_TEXT = "rejected"
    # ACCEPTED_COLOR = "green"
    # ACCEPTED_TEXT = "accepted"
    # idx_accepted and idx_rejected should be disjoint lists coming from CaImAn. (0-indexing)
    # set to 1 all the entries that correspond to accepted components. Rest is 0.
    cell_category_original = [0 for i in range(len(idx_accepted) + len(idx_rejected))]
    for i_accepted in idx_accepted:
        cell_category_original[i_accepted] = 1
    cell_category_new = cell_category_original.copy()

    colormap = mpl.cm.get_cmap(cmap)
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape

    nA2 = (
        np.ravel(np.power(A, 2).sum(0))
        if isinstance(A, np.ndarray)
        else np.ravel(A.power(2).sum(0))
    )
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        # Y_r = np.array(spdiags(old_div(1, nA2), 0, nr, nr) *
        #               (A.T * np.matrix(Yr) -
        #                (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) -
        #                A.T.dot(A) * np.matrix(C)) + C)
        raise NotImplementedError("YrA is None; this has not been implemented yet")
    else:
        Y_r = C + YrA

    x = np.arange(T)
    if image_neurons is None:
        raise NotImplementedError(
            "image_neurons is None; this has not been implemented yet"
        )
        # image_neurons = A.mean(1).reshape((d1, d2), order='F')

    coors = get_contours(A, (d1, d2), thr)
    cc1 = [cor["coordinates"][:, 0] for cor in coors]
    cc2 = [cor["coordinates"][:, 1] for cor in coors]
    c1 = cc1[0]
    c2 = cc2[0]

    # split sources up, such that Bokeh does not warn
    # "ColumnDataSource's columns must be of the same length"
    # contains traces of single neuron
    source = ColumnDataSource(data=dict(x=x, y=Y_r[0] / 100, y2=C[0] / 100))
    # contains all traces; use this to update source
    source_ = ColumnDataSource(data=dict(z=Y_r / 100, z2=C / 100))
    source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
    source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))
    categories = ColumnDataSource(data=dict(cats=cell_category_original))
    categories_new = ColumnDataSource(data=dict(cats=cell_category_new))
    # TODO: create list that contains the neurons the slide can go over, mapping slider index (1 to N) to neuron index
    #       in source.  Depending on dropdown setting, re-make this list to include only accepted, only rejected, all,
    #       or modified-only components.
    neurons_to_show = ColumnDataSource(
        data=dict(idx=[i for i in range(len(cell_category_original))])
    )
    slider_code = """
            var data = source.data
            var data_ = source_.data
            var indices = neurons_to_show.data['idx'];  // map neuron indices (in source) to slider values
            var f = cb_obj.value - 1  // slider value. (converted to 0-indexing)
            var x = data['x']
            var y = data['y']
            var y2 = data['y2']
            // update source (i.e. single neuron trace) to the currently selected neuron
            for (var i = 0; i < x.length; i++) {
                y[i] = data_['z'][i+indices[f]*x.length]
                y2[i] = data_['z2'][i+indices[f]*x.length]
            }
            // update category
            var cats = categories.data["cats"];
            var cats_new = categories_new.data["cats"];

            var data2_ = source2_.data;
            var data2 = source2.data;
            var c1 = data2['c1'];
            var c2 = data2['c2'];
            var cc1 = data2_['cc1'];
            var cc2 = data2_['cc2'];
            for (var i = 0; i < c1.length; i++) {
                   c1[i] = cc1[indices[f]][i]
                   c2[i] = cc2[indices[f]][i]
            }
            console.log(cats[f]);
            console.log(cats[f] > 0);
            console.log(indices[f]);
            // update button text and color
            btn_idx.label= "#" + String(indices[f]+1);  // Keep 1-indexing for showing neurons
            if (cats[indices[f]] > 0) {
                btn_orig_cat.label = 'Original: accepted';
                btn_orig_cat.background = 'green';
            }
            else {
                btn_orig_cat.label = 'Original: rejected';
                btn_orig_cat.background = 'red';
            }
            if (cats_new[indices[f]] > 0) {
                btn_new_cat.label = 'Current: accepted';
                btn_new_cat.background = 'green';
            }
            else {
                btn_new_cat.label = 'Current: rejected';
                btn_new_cat.background = 'red';
            }
            source2.change.emit();
            source.change.emit();
        """

    if r_values is not None:
        slider_code += """
            var mets = metrics.data['mets']
            mets[1] = metrics_.data['R'][indices[f]].toFixed(3)
            mets[2] = metrics_.data['SNR'][indices[f]].toFixed(3)
            metrics.change.emit();
        """
        metrics = ColumnDataSource(
            data={
                "y": (3, 2, 1, 0),
                "mets": (
                    "",
                    f"{r_values[0]:7.3f} ",
                    f"{SNR[0]:7.3f}",
                    (
                        "N/A"
                        if np.sum(cnn_preds) in (0, None)
                        else f"{cnn_preds[0]:7.3f}"
                    ),
                ),
                "keys": ("Evaluation Metrics", "Spatial corr:", "SNR:", "CNN:"),
            }
        )
        if np.sum(cnn_preds) in (0, None):
            metrics_ = ColumnDataSource(data=dict(R=r_values, SNR=SNR))
        else:
            metrics_ = ColumnDataSource(data=dict(R=r_values, SNR=SNR, CNN=cnn_preds))
            slider_code += """
                mets[3] = metrics_.data['CNN'][indices[f]].toFixed(3)
            """
        labels = LabelSet(x=0, y="y", text="keys", source=metrics, render_mode="canvas")
        labels2 = LabelSet(
            x=10,
            y="y",
            text="mets",
            source=metrics,
            render_mode="canvas",
            text_align="right",
        )
        plot2 = bpl.figure(plot_width=200, plot_height=100, toolbar_location=None)
        plot2.axis.visible = False
        plot2.grid.visible = False
        plot2.tools.visible = False
        plot2.line([0, 10], [0, 4], line_alpha=0)
        plot2.add_layout(labels)
        plot2.add_layout(labels2)
    else:
        metrics, metrics_ = None, None
    btn_idx = Button(
        label="#" + str(neurons_to_show.data["idx"][0] + 1), disabled=True, width=60
    )
    # btn_idx = Button(label="#", disabled=True, width=60)

    original_status = Button(
        label=(
            "original: accepted"
            if cell_category_original[0] > 0
            else "original: rejected"
        ),
        disabled=True,
        width=120,
        background="green" if cell_category_original[0] > 0 else "red",
    )
    current_status = Button(
        label="current: accepted" if cell_category_new[0] > 0 else "current: rejected",
        disabled=True,
        width=120,
        background="green" if cell_category_new[0] > 0 else "red",
    )

    callback = CustomJS(
        args=dict(
            source=source,
            source_=source_,
            source2=source2,
            source2_=source2_,
            metrics=metrics,
            metrics_=metrics_,
            categories=categories,
            categories_new=categories_new,
            btn_idx=btn_idx,
            btn_orig_cat=original_status,
            btn_new_cat=current_status,
            neurons_to_show=neurons_to_show,
        ),
        code=slider_code,
    )

    # TODO: start adding parameters to slider_callback, see when it breaks down.
    # TODO: callback does not seem to work! No log print...

    # plot = bpl.figure(plot_width=600, plot_height=200, x_range=Range1d(0, Y_r.shape[1]))
    # plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    # if denoised_color is not None:
    #     plot.line('x', 'y2', source=source, line_width=1,
    #               line_alpha=0.6, color=denoised_color)

    xr = Range1d(start=0, end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0], end=0)
    # plot1 = bpl.figure(x_range=xr, y_range=yr,
    #                    plot_width=int(min(1, d2 / d1) * 300),
    #                    plot_height=int(min(1, d1 / d2) * 300))
    #
    # plot1.image(image=[image_neurons[::-1, :]], x=0,
    #             y=image_neurons.shape[0], dw=d2, dh=d1, palette=grayp)
    # plot1.patch('c1', 'c2', alpha=0.6, color='purple',
    #             line_width=2, source=source2)

    transfer_button = Button(label="Transfer", width=70)
    save_button = Button(label="Save changes", width=80)

    menu = [
        ("Rejected", "rejected"),
        ("Accepted", "accepted"),
        ("All", "all"),
        ("Modified", "modified"),
    ]
    dropdown = Dropdown(
        label="Show " + menu[cell_category_original[2]][1],
        button_type="warning",
        menu=menu,
        width=100,
        name="dropdown",
    )  # default is to show all

    slider = Slider(start=1, end=Y_r.shape[0], value=1, step=1, title="Neuron Number")

    if not (Y_r.shape[0] > 1):
        # bpl.show(row(plot1 if r_values is None else
        #              column(plot1, plot2), plot))
        raise NotImplementedError(
            "Y_r.shape[0] !> 1. This case has not been implemented yet."
        )

    # slider.js_on_change('value', slider_callback)  # FIXME: this line of code is messing up the whole function.
    #  What is wrong with callback?

    dropdown_code = """
            var cats_orig = categories.data['cats'];
            // current dropdown selection is item
            // Change label
            dropdown.label = 'Show ' + this.item;
            // Change slider values
            if (this.item == 'accepted') { // show originally accepted
                const n_accepted = categories.data['cats'].reduce((a, b) => a + b, 0);
                // Create an empty array for the indices of accepted components. array[i] = index of i-th accepted 
                neuron.
                var accepted_indices = [];
                accepted_indices.length = n_accepted; 
                accepted_indices.fill(0);
                // TODO: need to get list of indices in categories that are non-zero. Iterate through categories,
                // if element is non-zero, change next element in accepted_indices to the value. Increment 
                accepted_indices pointer.
                // If this kind of rebuilding is too slow, can create more data sources, and change them every time 
                we change neuron classification.
                var i_current = 0; // pointer to first  empty position in accepted_indices 
                for (var i = 0; i < cats_orig.length; i++) {
                    if (cats_orig[i] > 0) { // the component was accepted originally 
                        accepted_indices[i_current] = i;
                        i_current += 1; 
                    }
                }
                console.log("Show accepted");
                console.log(neurons_to_show.data['idx'].length);
                neurons_to_show.data['idx'] = accepted_indices;
                console.log(neurons_to_show.data['idx'].length);
                slider.end = neurons_to_show.data['idx'].length;
            }
            else if (this.item == 'rejected') {
                const n_rejected = cats_orig.length - categories.data['cats'].reduce((a, b) => a + b, 0);

                var rejected_indices = [];
                rejected_indices.length = n_rejected; 
                rejected_indices.fill(0);
                var i_current = 0; // pointer to first  empty position in rejected_indices 
                for (var i = 0; i < cats_orig.length; i++) {
                    if (cats_orig[i] == 0) { // the component was rejected originally 
                        rejected_indices[i_current] = i;
                        i_current += 1; 
                    }
                }
                console.log("Show rejected");
                console.log(neurons_to_show.data['idx'].length);
                neurons_to_show.data['idx'] = rejected_indices;
                console.log(neurons_to_show.data['idx'].length);
                slider.end = neurons_to_show.data['idx'].length;
            }
            else if (this.item == 'modified') {
                var cats_new = categories_new.data['cats'];
                // TODO: get number of cats_orig = cats_new, then get those components.
                //TODO: do not look at cat_new but the temporary value that will be saved to file.
                var n_modified = 0;
                var modified_indices = [];
                for (var i = 0; i < cats_orig.length; i++){
                    if (cats_orig[i] != cats_new[i]) {
                        n_modified++;
                        modified_indices.push(i);
                    }
                }
                if (n_modified > 0){
                    console.log("Show modified");
                    console.log(neurons_to_show.data['idx'].length);
                    neurons_to_show.data['idx'] = modified_indices;
                    console.log(neurons_to_show.data['idx'].length);
                    slider.end = neurons_to_show.data['idx'].length;
                    }
            }
            else { // show all components
                var all_indices = [];
                all_indices.length = cats_orig.length; 
                all_indices.fill(0);  // TODO: probably possible to replace loop below with function in fill()
                for (var i = 0; i < cats_orig.length; i++) {
                    all_indices[i] = i;
                }
                console.log("Show all");
                console.log(neurons_to_show.data['idx'].length);
                neurons_to_show.data['idx'] = all_indices;
                console.log(neurons_to_show.data['idx'].length);
                slider.end = neurons_to_show.data['idx'].length;
            }


            """
    dropdown_callback = CustomJS(
        args=dict(
            dropdown=dropdown,
            slider=slider,
            categories=categories,
            categories_new=categories_new,
            neurons_to_show=neurons_to_show,
        ),
        code=dropdown_code,
    )
    dropdown.js_on_event("menu_item_click", dropdown_callback)

    # on pressing transfer, change the current category of the neuron.
    on_transfer_pressed = CustomJS(
        args={
            "transfer_button": transfer_button,
            "curr_cat": categories_new,
            "btn_curr_cat": current_status,
            "slider": slider,
        },
        code="""
    var i_cell = slider.value - 1
    var cats_new = curr_cat.data['cats'];
    console.log(String(i_cell));
    // change current category
    if (cats_new[i_cell] > 0) { // currently accepted -> change to rejected
        cats_new[i_cell] = 0;
        btn_curr_cat.label = 'current: rejected';
        btn_curr_cat.background = 'red';
    }
    else {  // currently rejected -> change to accepted
        cats_new[i_cell] = 1;
        btn_curr_cat.label = 'current: accepted';
        btn_curr_cat.background = 'green';
    }
    // cats_new.change.emit();  //change is undefined here
    """,
    )

    out_fname = get_filename_with_date("manual_classification", extension=".txt")
    save_data_callback = CustomJS(
        args={"new_cats": categories_new, "out_fname": out_fname},
        code="""
        var data = new_cats.data['cats'];
        var out = "";
        for (var i=0; i < data.length; i++) {
            out += data[i];
            out += " ";
        }
        var file = new Blob([out], {type: 'text/plain'});
        var elem = window.document.createElement('a');
        elem.href = window.URL.createObjectURL(file);
        elem.download = out_fname;
        document.body.appendChild(elem);
        elem.click();
        document.body.removeChild(elem);
        """,
    )

    transfer_button.js_on_click(on_transfer_pressed)
    save_button.js_on_click(save_data_callback)
    # bpl.show(layout([[slider, transfer_button, btn_idx, original_status, current_status, dropdown, save_button],
    #                  row(plot1 if r_values is None else column(plot1, plot2), plot)]))
    bpl.show(
        layout(
            [
                [
                    slider,
                    transfer_button,
                    btn_idx,
                    original_status,
                    current_status,
                    dropdown,
                    save_button,
                ]
            ]
        )
    )
    # return Y_r

    # TODO: create save button to write results to a txt file. See
    #  https://stackoverflow.com/questions/54215667/bokeh-click-button-to-save-widget-values-to-txt-file-using
    #  -javascript
    # and https://stackoverflow.com/questions/62290866/python-bokeh-applicationunable-to-export-updated-data-from
    # -webapp-to-local-syst
    return out_fname


def nb_view_patches_manual_control(
    Yr,
    A,
    C,
    b,
    f,
    d1,
    d2,
    YrA=None,
    image_neurons=None,
    thr=0.99,
    denoised_color=None,
    cmap="jet",
    r_values=None,
    SNR=None,
    cnn_preds=None,
    mode: str = None,
    n_neurons: int = 0,
    idx=None,
):
    """
    Interactive plotting utility for ipython notebook
    Args:
        Yr: np.ndarray
            movie

        A,C,b,f: np.ndarrays
            outputs of matrix factorization algorithm

        d1,d2: floats
            dimensions of movie (x and y)

        YrA:   np.ndarray
            ROI filtered residual as it is given from update_temporal_components
            If not given, then it is computed (K x T)

        image_neurons: np.ndarray
            image to be overlaid to neurons (for instance the average)

        thr: double
            threshold regulating the extent of the displayed patches

        denoised_color: string or None
            color name (e.g. 'red') or hex color code (e.g. '#F0027F')

        cmap: string
            name of colormap (e.g. 'viridis') used to plot image_neurons
            :param r_values:

        mode: string
            The initial view mode to show. It should be one of 'accepted', 'rejected', 'all'. The fourth option,
            'modified', would cause an empty plot.

        idx: List
            The idx_components or idx_components_bad field of the estimates object.

    """
    # No easy way to use these in CustomJS. Could define beginning of variable 'code' like this, and append the rest
    # REJECTED_COLOR = "red"
    # REJECTED_TEXT = "rejected"
    # ACCEPTED_COLOR = "green"
    # ACCEPTED_TEXT = "accepted"

    # idx_accepted and idx_rejected should be disjoint lists coming from CaImAn. (0-indexing)
    # set to 1 all the entries that correspond to accepted components. Rest is 0.
    if mode == "rejected":
        orig_cat = 0
        CAT_COLOR = "red"
    else:
        orig_cat = 1
        CAT_COLOR = "green"

    cell_category_new = [orig_cat for i in range(n_neurons)]

    colormap = mpl.cm.get_cmap(cmap)
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape

    nA2 = (
        np.ravel(np.power(A, 2).sum(0))
        if isinstance(A, np.ndarray)
        else np.ravel(A.power(2).sum(0))
    )
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        # Y_r = np.array(spdiags(old_div(1, nA2), 0, nr, nr) *
        #               (A.T * np.matrix(Yr) -
        #                (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) -
        #                A.T.dot(A) * np.matrix(C)) + C)
        raise NotImplementedError("YrA is None; this has not been implemented yet")
    else:
        Y_r = C + YrA

    x = np.arange(T)
    if image_neurons is None:
        raise NotImplementedError(
            "image_neurons is None; this has not been implemented yet"
        )
        # image_neurons = A.mean(1).reshape((d1, d2), order='F')

    coors = get_contours(A, (d1, d2), thr)
    cc1 = [cor["coordinates"][:, 0] for cor in coors]
    cc2 = [cor["coordinates"][:, 1] for cor in coors]
    c1 = cc1[0]
    c2 = cc2[0]

    # split sources up, such that Bokeh does not warn
    # "ColumnDataSource's columns must be of the same length"
    # contains traces of single neuron
    source = ColumnDataSource(data=dict(x=x, y=Y_r[0] / 100, y2=C[0] / 100))
    # contains all traces; use this to update source
    source_ = ColumnDataSource(data=dict(z=Y_r / 100, z2=C / 100))
    source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
    source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))

    categories_new = ColumnDataSource(data=dict(cats=cell_category_new))
    index_map = ColumnDataSource(data=dict(indices=idx))
    # make original category accessible to javascript
    category_original = ColumnDataSource(
        data=dict(cat=[0 if mode == "rejected" else 1])
    )

    slider_code = """
            var data = source.data;
            var data_ = source_.data;
            var f = cb_obj.value - 1;  //Switch to zero-indexing from displayed 1-indexing
            var x = data['x'];
            var y = data['y'];
            var y2 = data['y2'];
            var cats_new=categories_new.data['cats'];
            var indices = index_map.data['indices'];
            
            for (var i = 0; i < x.length; i++) {
                y[i] = data_['z'][i+f*x.length];
                y2[i] = data_['z2'][i+f*x.length];
            }

            var data2_ = source2_.data;
            var data2 = source2.data;
            var c1 = data2['c1'];
            var c2 = data2['c2'];
            var cc1 = data2_['cc1'];
            var cc2 = data2_['cc2'];

            for (var i = 0; i < c1.length; i++) {
                   c1[i] = cc1[f][i];
                   c2[i] = cc2[f][i];
            }
            btn_idx.label = '# ' + indices[f];
            
            if (cats_new[f] > 0) {
                btn_current_status.label = 'Current: accepted';
                btn_current_status.background = 'green';
            }
            else {
                btn_current_status.label = 'Current: rejected';
                btn_current_status.background = 'red';
            }
            
            source2.change.emit();
            source.change.emit();
        """

    if r_values is not None:
        slider_code += """
            var mets = metrics.data['mets'];
            mets[1] = metrics_.data['R'][f].toFixed(3);
            mets[2] = metrics_.data['SNR'][f].toFixed(3);
            metrics.change.emit();
        """
        metrics = ColumnDataSource(
            data=dict(
                y=(3, 2, 1, 0),
                mets=(
                    "",
                    f"{r_values[0]:7.3f}",
                    f"{SNR[0]:7.3f}",
                    (
                        "N/A"
                        if np.sum(cnn_preds) in (0, None)
                        else f"{cnn_preds[0]:7.3f}"
                    ),
                ),
                keys=("Evaluation Metrics", "Spatial corr:", "SNR:", "CNN:"),
            )
        )
        if np.sum(cnn_preds) in (0, None):
            metrics_ = ColumnDataSource(data={"R": r_values, "SNR": SNR})
        else:
            metrics_ = ColumnDataSource(
                data={"R": r_values, "SNR": SNR, "CNN": cnn_preds}
            )
            slider_code += """
                mets[3] = metrics_.data['CNN'][f].toFixed(3)
            """
        labels = LabelSet(x=0, y="y", text="keys", source=metrics, render_mode="canvas")
        labels2 = LabelSet(
            x=10,
            y="y",
            text="mets",
            source=metrics,
            render_mode="canvas",
            text_align="right",
        )
        plot2 = bpl.figure(plot_width=200, plot_height=100, toolbar_location=None)
        plot2.axis.visible = False
        plot2.grid.visible = False
        plot2.tools.visible = False
        plot2.line([0, 10], [0, 4], line_alpha=0)
        plot2.add_layout(labels)
        plot2.add_layout(labels2)
    else:
        metrics, metrics_ = None, None

    plot = bpl.figure(plot_width=600, plot_height=200, x_range=Range1d(0, Y_r.shape[1]))
    plot.line("x", "y", source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line(
            "x", "y2", source=source, line_width=1, line_alpha=0.6, color=denoised_color
        )

    xr = Range1d(start=0, end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0], end=0)
    plot1 = bpl.figure(
        x_range=xr,
        y_range=yr,
        plot_width=int(min(1, d2 / d1) * 300),
        plot_height=int(min(1, d1 / d2) * 300),
    )

    plot1.image(
        image=[image_neurons[::-1, :]],
        x=0,
        y=image_neurons.shape[0],
        dw=d2,
        dh=d1,
        palette=grayp,
    )
    plot1.patch("c1", "c2", alpha=0.6, color="purple", line_width=2, source=source2)

    slider = Slider(
        start=1, end=Y_r.shape[0], value=1, step=1, title=mode + " neuron number"
    )
    btn_current_status = Button(
        label="current: accepted" if cell_category_new[0] > 0 else "current: rejected",
        disabled=True,
        width=120,
        background="green" if cell_category_new[0] > 0 else "red",
    )
    btn_transfer = Button(label="Transfer", width=70)
    btn_save = Button(label="Save changes", width=80)
    btn_idx = Button(
        label="# " + str(index_map.data["indices"][0]),
        disabled=True,
        width=60,
        background=CAT_COLOR,
    )

    out_fname = get_filename_with_date("manual_class_" + mode, extension=".txt")

    callback = CustomJS(
        args=dict(
            source=source,
            source_=source_,
            source2=source2,
            source2_=source2_,
            metrics=metrics,
            metrics_=metrics_,
            btn_current_status=btn_current_status,
            categories_new=categories_new,
            index_map=index_map,
            btn_idx=btn_idx,
        ),
        code=slider_code,
    )
    # on pressing transfer, change the current category of the neuron.
    on_transfer_pressed = CustomJS(
        args={
            "curr_cat": categories_new,
            "btn_curr_cat": btn_current_status,
            "slider": slider,
        },
        code="""
           var i_cell = slider.value - 1
           var cats_new = curr_cat.data['cats'];
           console.log(String(i_cell));
           // change current category
           if (cats_new[i_cell] > 0) { // currently accepted -> change to rejected
               cats_new[i_cell] = 0;
               btn_curr_cat.label = 'current: rejected';
               btn_curr_cat.background = 'red';
           }
           else {  // currently rejected -> change to accepted
               cats_new[i_cell] = 1;
               btn_curr_cat.label = 'current: accepted';
               btn_curr_cat.background = 'green';
           }
           """,
    )

    save_data_callback = CustomJS(
        args={
            "new_cats": categories_new,
            "index_map": index_map,
            "out_fname": out_fname,
            "category_original": category_original,
        },
        code="""
        const cat_orig = category_original.data['cat'];
        var neuron_index = index_map.data['indices'];
        var cat_binary = new_cats.data['cats'];
        var out = "";
        for (var i=0; i < cat_binary.length; i++) {
            if (cat_binary[i] != cat_orig) {
                out += neuron_index[i]
                out += '\\n';
            }
        }
        var file = new Blob([out], {type: 'text/plain'});
        var elem = window.document.createElement('a');
        elem.href = window.URL.createObjectURL(file);
        elem.download = out_fname;
        document.body.appendChild(elem);
        elem.click();
        document.body.removeChild(elem);
        """,
    )

    slider.js_on_change("value", callback)
    btn_transfer.js_on_click(on_transfer_pressed)
    btn_save.js_on_click(save_data_callback)
    if not (Y_r.shape[0] > 1):
        # bpl.show(row(plot1 if r_values is None else
        #              column(plot1, plot2), plot))
        raise NotImplementedError(
            "Y_r.shape[0] !> 1. This case has not been implemented yet."
        )

    bpl.show(
        layout(
            [
                [slider, btn_idx, btn_transfer, btn_current_status, btn_save],
                row(plot1 if r_values is None else column(plot1, plot2), plot),
            ]
        )
    )
    # return Y_r

    # TODO: create save button to write results to a txt file. See
    #  https://stackoverflow.com/questions/54215667/bokeh-click-button-to-save-widget-values-to-txt-file-using
    #  -javascript
    # and https://stackoverflow.com/questions/62290866/python-bokeh-applicationunable-to-export-updated-data-from
    # -webapp-to-local-syst
    return out_fname


def nb_view_components_with_lfp_movement(
    estimates,
    t_lfp: np.array = None,
    y_lfp: np.array = None,
    t_mov: np.array = None,
    y_mov: np.array = None,
    Yr=None,
    img=None,
    idx=None,
    denoised_color=None,
    cmap="jet",
    thr=0.99,
):
    """view spatial and temporal components interactively in a notebook, along with LFP and movement

    Args:
        estimates : the estimates attribute of a CNMF instance
        t_lfp: np.ndarray
            time data of lfp recording
        y_lfp: np.ndarray
            amplitude of lfp recording
        t_mov: np.ndarray
            time data of movement recording
        y_mov: np.ndarray
            amplitude of movement recording
        Yr :    np.ndarray
            movie in format pixels (d) x frames (T)

        img :   np.ndarray
            background image for contour plotting. Default is the mean
            image of all spatial components (d1 x d2)

        idx :   list
            list of components to be plotted

        thr: double
            threshold regulating the extent of the displayed patches

        denoised_color: string or None
            color name (e.g. 'red') or hex color code (e.g. '#F0027F')

        cmap: string
            name of colormap (e.g. 'viridis') used to plot image_neurons
            :param estimates:
    """
    if "csc_matrix" not in str(type(estimates.A)):
        estimates.A = scipy.sparse.csc_matrix(estimates.A)

    plt.ion()
    nr, T = estimates.C.shape
    if estimates.R is None:
        estimates.R = estimates.YrA
    if estimates.R.shape != [nr, T]:
        if estimates.YrA is None:
            estimates.compute_residuals(Yr)
        else:
            estimates.R = estimates.YrA

    if img is None:
        img = np.reshape(np.array(estimates.A.mean(axis=1)), estimates.dims, order="F")

    if idx is None:
        nb_view_patches_with_lfp_movement(
            Yr,
            estimates.A,
            estimates.C,
            estimates.b,
            estimates.f,
            estimates.dims[0],
            estimates.dims[1],
            t_lfp=t_lfp,
            y_lfp=y_lfp,
            t_mov=t_mov,
            y_mov=y_mov,
            YrA=estimates.R,
            image_neurons=img,
            thr=thr,
            denoised_color=denoised_color,
            cmap=cmap,
            r_values=estimates.r_values,
            SNR=estimates.SNR_comp,
            cnn_preds=estimates.cnn_preds,
        )
    else:
        nb_view_patches_with_lfp_movement(
            Yr,
            estimates.A.tocsc()[:, idx],
            estimates.C[idx],
            estimates.b,
            estimates.f,
            estimates.dims[0],
            estimates.dims[1],
            t_lfp=t_lfp,
            y_lfp=y_lfp,
            t_mov=t_mov,
            y_mov=y_mov,
            YrA=estimates.R[idx],
            image_neurons=img,
            thr=thr,
            denoised_color=denoised_color,
            cmap=cmap,
            r_values=None if estimates.r_values is None else estimates.r_values[idx],
            SNR=None if estimates.SNR_comp is None else estimates.SNR_comp[idx],
            cnn_preds=(
                None
                if np.sum(estimates.cnn_preds) in (0, None)
                else estimates.cnn_preds[idx]
            ),
        )
    return estimates


def nb_view_components_manual_control(
    estimates,
    Yr=None,
    img=None,
    idx=None,
    denoised_color=None,
    cmap="jet",
    thr=0.99,
    mode: str = "rejected",
):
    """view spatial and temporal components interactively in a notebook
    Sadly, does not work due to ??? (Neither Javascript nor Python can not say why. At some point, there is an overflow,
    probably.
    Args:
        estimates : the estimates attribute of a CNMF instance
        t_lfp: np.ndarray
            time data of lfp recording
        y_lfp: np.ndarray
            amplitude of lfp recording
        t_mov: np.ndarray
            time data of movement recording
        y_mov: np.ndarray
            amplitude of movement recording
        Yr :    np.ndarray
            movie in format pixels (d) x frames (T)

        img :   np.ndarray
            background image for contour plotting. Default is the mean
            image of all spatial components (d1 x d2)

        idx :   list
            list of components to be plotted

        thr: double
            threshold regulating the extent of the displayed patches

        denoised_color: string or None
            color name (e.g. 'red') or hex color code (e.g. '#F0027F')

        cmap: string
            name of colormap (e.g. 'viridis') used to plot image_neurons
            :param estimates:

                mode: string, one of ["all", "rejected", "accepted"]  # "modified" is also a category but it would be empty
                        Whether to go through accepted components and reject manually ("accepted" or reject"), or go through
                        rejected components and move manually to accepted ("rejected" or "accept").
    """
    from matplotlib import pyplot as plt
    import scipy

    # TODO: if refit is used, estimates.idx_components and idx_components_bad are empty (None). Need to still plot
    #  these as all accepted
    if "csc_matrix" not in str(type(estimates.A)):
        estimates.A = scipy.sparse.csc_matrix(estimates.A)

    if hasattr(estimates, "idx_components"):
        if estimates.idx_components is not None:
            idx_accepted = estimates.idx_components
    else:
        raise Exception("estimates does not have idx_components field")
    if hasattr(estimates, "idx_components_bad"):
        if estimates.idx_components_bad is not None:
            idx_rejected = estimates.idx_components_bad
    else:
        raise Exception("estimates does not have idx_components_bad field")
    plt.ion()
    nr, T = estimates.C.shape
    if estimates.R is None:
        estimates.R = estimates.YrA
    if estimates.R.shape != [nr, T]:
        if estimates.YrA is None:
            estimates.compute_residuals(Yr)
        else:
            estimates.R = estimates.YrA

    if img is None:
        img = np.reshape(np.array(estimates.A.mean(axis=1)), estimates.dims, order="F")
    # FIXME: unfortunately, it is impossible to plot all components... Probable cause is overflow error. This might also
    # occur if the number of accepted or rejected components is too high. Maybe later, this limitation will be solved.
    if mode == "rejected":
        idx = estimates.idx_components_bad
    elif mode == "accepted":
        idx = estimates.idx_components
    else:
        raise NotImplementedError(
            "Only accepted and rejected modes are supported. The reason for lack of showing all components, "
            "for example, is a limitation in javascript."
        )
    n_neurons = len(idx)
    out_fname = nb_view_patches_manual_control(
        Yr,
        estimates.A.tocsc()[:, idx],
        estimates.C[idx],
        estimates.b,
        estimates.f,
        estimates.dims[0],
        estimates.dims[1],
        YrA=estimates.R[idx],
        image_neurons=img,
        thr=thr,
        denoised_color=denoised_color,
        cmap=cmap,
        r_values=None if estimates.r_values is None else estimates.r_values[idx],
        SNR=None if estimates.SNR_comp is None else estimates.SNR_comp[idx],
        cnn_preds=(
            None
            if np.sum(estimates.cnn_preds) in (0, None)
            else estimates.cnn_preds[idx]
        ),
        mode=mode,
        n_neurons=n_neurons,
        idx=idx,
    )
    # return estimates
    return out_fname


def reopen_manual_control(fname: str, downloads_folder: str = None) -> List:
    """
    :param fname: the file name parameter of nb_view_components_manual_control
    :return: list where each element is 0 or 1, corresponding to whether neuron
    i is rejected (0) or accepted (1) after manual inspection.
    """

    if downloads_folder is None:
        # often, this is the downloads folder, but not always
        if os.path.exists("D:\\Downloads"):
            downloads_folder = "D:\\Downloads"
            print(f"Found {downloads_folder}, assuming file is located here.")
        else:
            downloads_folder = open_dir("Find downloads directory.")
    changed_indices = []
    with open(os.path.join(downloads_folder, fname), "r", encoding="utf-8") as file:
        for line in file.readlines():
            changed_indices.append(int(line.rstrip()))
    return changed_indices
