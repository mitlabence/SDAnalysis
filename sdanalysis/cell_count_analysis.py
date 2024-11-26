"""
cell_count_analysis.py - This script is used to analyze the number of cells in each recording of a
specified dataset.
"""

import os
import json
import argparse
import h5py
import pandas as pd
import numpy as np
import data_documentation as dd
import custom_io as cio
import env_reader


def create_results(
    dict_n_cells: dict,
    dict_n_frames: dict,
    dict_mouse_colors: dict,
    data_documentation: dd.DataDocumentation,
) -> pd.DataFrame:
    """
    Create a DataFrame with the results of the cell count analysis.
    Args:
        dict_n_cells (dict): Dictionary with following format:
        {mouse_id:
            {exp_type:
                {"pre": [n_cells for each recording pre stage],
                "post": [n_cells for each recording post stage],
                "uuid": [corresponding uuids]}
            }
        }
        dict_n_frames (dict): Dictionary with following format:
        {mouse_id:
            {exp_type:
                {"pre": [n_frames for each recording pre stage],
                "post": [n_frames for each recording post stage],
                "uuid": [corresponding uuids]}
            }
        }

        dict_mouse_colors (dict): Dictionary with mouse_id as key and color as value
        data_documentation (dd.DataDocumentation): DataDocumentation object
    Returns:
        pd.DataFrame: _description_
    """
    dict_results = {
        "mouse_id": [],
        "exp_type": [],
        "uuid": [],
        "cell_count_pre": [],
        "cell_count_post": [],
        "n_frames_pre": [],
        "n_frames_post": [],
        "color": [],
    }

    for mouse_id, exp_types_for_mouse in dict_n_cells.items():
        for exp_type, n_cells_pre_post in exp_types_for_mouse.items():
            color = dict_mouse_colors[mouse_id]
            n_cells_pre = n_cells_pre_post["pre"]
            n_cells_post = n_cells_pre_post["post"]
            mouse_ids = [mouse_id] * len(n_cells_pre)
            exp_types = [exp_type] * len(n_cells_post)

            dict_results["mouse_id"].extend(mouse_ids)
            dict_results["exp_type"].extend(exp_types)
            dict_results["uuid"].extend(n_cells_pre_post["uuid"])
            dict_results["cell_count_pre"].extend(n_cells_pre)
            dict_results["cell_count_post"].extend(n_cells_post)
            dict_results["n_frames_pre"].extend(
                dict_n_frames[mouse_id][exp_type]["pre"]
            )
            dict_results["n_frames_post"].extend(
                dict_n_frames[mouse_id][exp_type]["post"]
            )
            dict_results["color"].extend([color] * len(n_cells_pre))

    df_results = pd.DataFrame(dict_results)
    df_results = df_results.sort_values(by=["mouse_id", "exp_type", "uuid"])
    df_results["post_pre_ratio"] = (
        df_results["cell_count_post"] / df_results["cell_count_pre"]
    )
    # add stim duration (if exists)
    # TODO: only works if uuid == recording_uuid, not event_uuid. (i.e. only for sessions
    # spanning one recording)
    df_results["stim_duration"] = df_results.apply(
        lambda row: _get_stim_duration_or_nan(row["uuid"], data_documentation), axis=1
    )
    # reorganize columns
    return df_results[
        [
            "mouse_id",
            "exp_type",
            "uuid",
            "cell_count_pre",
            "cell_count_post",
            "color",
            "n_frames_pre",
            "n_frames_post",
            "stim_duration",
            "post_pre_ratio",
        ]
    ]


def _get_stim_duration_or_nan(
    uuid: str, data_documentation: dd.DataDocumentation
) -> float:
    """
    Get the duration of the stimulation for a given recording uuid.
    If the uuid is not found in the data documentation, return np.nan.

    Args:
        uuid (str): The uuid of the recording
        data_documentation (dd.DataDocumentation): The data documentation object

    Returns:
        float: The duration of the stimulation in seconds, or np.nan if the uuid is not found
    """
    try:
        return data_documentation.get_stim_duration_for_uuid(uuid)
    except IndexError:
        return np.nan


def get_dataset_type(dict_n_cells: dict) -> str:
    """
    Get the type of the dataset based on the keys of the dictionary

    Args:
        dict_n_cells (dict): Dictionary with following format:
        {mouse_id:
            {exp_type:
                {"pre": [n_cells for each recording pre stage],
                "post": [n_cells for each recording post stage],
                "uuid": [corresponding uuids]}
            }
        }
    Returns:
        str: "tmev" (if all of the recordings' experiment type contains "tmev") or "stim".
    Raises:
        ValueError: If the dataset type is not "tmev" or "stim" (i.e. a mixture)
    """
    contains_tmev = False
    contains_stim = False  # avoid false "tmev" if only last experiment being tmev
    for _, dict_exp_type in dict_n_cells.items():
        for exp_type in dict_exp_type:
            if "tmev" in exp_type:
                contains_tmev = True
            else:  # anything other than "tmev" is considered "stim" now
                contains_stim = True
    if contains_stim and contains_tmev:
        raise ValueError("Dataset type cannot be determined.")
    return "tmev" if contains_tmev else "stim"


def extract_cell_count_from_files(dict_fpaths: dict) -> dict:
    """
    Extract the number of cells from the files and return a dictionary.

    Args:
        dict_fpaths (dict): Dictionary containing the file paths of the data to open with format:
        {mouse_id:
            {exp_type:
                {"pre": [fpaths for each recording pre stage],
                "post": [fpaths for each recording post stage]}
            }
        }
        data_documentation (dd.DataDocumentation): The data documentation

    Returns:
        dict: The resulting dataset with the following format:
        {mouse_id:
            {exp_type:
                {"pre": [n_cells for each recording pre stage],
                "post": [n_cells for each recording post stage],
                "uuid": [corresponding uuids]}
            }
        }
    """
    # {mouse_id: {exp_type: {"pre": [n_cells], "post": [n_cells], "uuid": [uuids]}}}
    dict_n_cells = {}
    for mouse_id in dict_fpaths:
        dict_n_cells[mouse_id] = {}
        for exp_type in dict_fpaths[mouse_id]:
            dict_n_cells[mouse_id][exp_type] = {}
            fpaths_pre = dict_fpaths[mouse_id][exp_type]["pre"]
            fpaths_post = dict_fpaths[mouse_id][exp_type]["post"]
            assert len(fpaths_pre) == len(fpaths_post)
            n_cells_pre = []
            n_cells_post = []
            uuids = []
            for fpath_pre, fpath_post in zip(fpaths_pre, fpaths_post):
                with h5py.File(fpath_pre, "r") as hdf_file:
                    n_cells_pre.append(hdf_file["estimates"]["C"].shape[0])
                with h5py.File(fpath_post, "r") as hdf_file:
                    n_cells_post.append(hdf_file["estimates"]["C"].shape[0])
                    uuid = hdf_file.attrs["uuid"]
                    uuids.append(uuid)
            dict_n_cells[mouse_id][exp_type]["pre"] = n_cells_pre
            dict_n_cells[mouse_id][exp_type]["post"] = n_cells_post
            dict_n_cells[mouse_id][exp_type]["uuid"] = uuids
    return dict_n_cells


def extract_frame_count_from_files(dict_fpaths: dict) -> dict:
    """
    Extract the number of frames from the files and return a dictionary.

    Args:
        dict_fpaths (dict): Dictionary containing the file paths of the data to open with format:
        {mouse_id:
            {exp_type:
                {"pre": [fpaths for each recording pre stage],
                "post": [fpaths for each recording post stage]}
            }
        }

    Returns:
        dict: The resulting dataset with the following format:
        {mouse_id:
            {exp_type:
                {"pre": [n_frames for each recording pre stage],
                "post": [n_frames for each recording post stage],
                "uuid": [corresponding uuids]}
            }
        }
    """
    dict_n_frames = {}
    for mouse_id in dict_fpaths:
        dict_n_frames[mouse_id] = {}
        for exp_type in dict_fpaths[mouse_id]:
            dict_n_frames[mouse_id][exp_type] = {}
            fpaths_pre = dict_fpaths[mouse_id][exp_type]["pre"]
            fpaths_post = dict_fpaths[mouse_id][exp_type]["post"]
            assert len(fpaths_pre) == len(fpaths_post)
            n_frames_pre = []
            n_frames_post = []
            uuids = []
            for fpath_pre, fpath_post in zip(fpaths_pre, fpaths_post):
                with h5py.File(fpath_pre, "r") as hdf_file:
                    n_frames_pre.append(hdf_file["estimates"]["C"].shape[1])
                with h5py.File(fpath_post, "r") as hdf_file:
                    n_frames_post.append(hdf_file["estimates"]["C"].shape[1])
                    uuids.append(hdf_file.attrs["uuid"])
            dict_n_frames[mouse_id][exp_type]["pre"] = n_frames_pre
            dict_n_frames[mouse_id][exp_type]["post"] = n_frames_post
            dict_n_frames[mouse_id][exp_type]["uuid"] = uuids
    return dict_n_frames


def load_pre_post_files(fpath_input: str, base_dir: str = None) -> dict:
    """
    Load the json file containing the (relative) file paths of the data to open. It is expected to
    have the form:
    {mouse_id:
        {exp_type:
            {"pre": [fpaths for each recording pre stage],
            "post": [fpaths for each recording post stage]}
        }
    }

    Args:
        fpath_input (str): The absolute file path of the json file
        base_dir (str): The base directory of the dataset. If None and the file paths in the json
        file are absolute, these file paths will not be changed. If None and the file paths are not
        absolute, an error will be raised.

    Returns:
        dict: The dictionary containing the file paths in same format as the input json file
    """
    # test the json file path
    if not (
        fpath_input is not None
        and os.path.splitext(fpath_input)[1] == ".json"
        and os.path.exists(fpath_input)
    ):
        raise ValueError("Invalid input file path")
    with open(fpath_input, "r", encoding="utf-8") as json_file:
        dict_fpaths = json.load(json_file)
    # test the root directory setting
    base_dir_available = base_dir is not None
    for mouse_id, exp_types in dict_fpaths.items():
        for exp_type, stages in exp_types.items():
            for stage, fpaths in stages.items():
                if (
                    not base_dir_available
                ):  # if root not given, need fpaths to be absolute
                    fpaths_are_absolute = [
                        os.path.isabs(fpath) for fpath in fpaths
                    ].all()
                    if not fpaths_are_absolute:
                        raise ValueError(
                            "No root directory given and the file paths are not absolute."
                        )
                else:
                    dict_fpaths[mouse_id][exp_type][stage] = [
                        os.path.join(base_dir, fpath) for fpath in fpaths
                    ]
    return dict_fpaths


def main(
    dict_input_file_paths: dict = None, save_results: bool = False, output_folder: str = None
) -> pd.DataFrame:
    """_summary_

    Args:
        dict_input_files (dict): dictionary that contains file paths of all data to
            open, with the following format:
            {mouse_id:
                {exp_type:
                    {"pre": [fpaths for each recording pre stage],
                    "post": [fpaths for each recording post stage]}
                }
            }
            The json file should be saved in the root directory of the dataset. I.e. if the file
            is at fpath\\json_file.json, then all file path fp in it will be assumed to be relative
            to fpath: fpath\\fp.
        save_results (bool): Whether to save the results to a file
        output_folder (str): The folder to save the output file.
            If save_results is False, this argument will be ignored.
            If save_results is True, the following checks will happen:
            1. if fpath_output is provided and it is a valid folder, the output file will be
                saved there.
            2. if fpath_output is provided but it is not a valid folder, an error will be raised.
            3. if fpath_output is not provided, the output file will be saved in the folder
            specified by the environment variable OUTPUT_FOLDER.
    """
    env_dict = env_reader.read_env()
    ddoc = dd.DataDocumentation.from_env_dict(env_dict)

    if save_results:
        if output_folder is None:
            if "OUTPUT_FOLDER" in env_dict:
                output_folder = env_dict["OUTPUT_FOLDER"]
            else:
                raise ValueError(
                    "Output folder not specified in argument or environment variable"
                )
        else:
            if not os.path.exists(output_folder):
                raise ValueError(f"Invalid output folder: {output_folder}")
        dtime = cio.get_datetime_for_fname()
    # load data
    dict_n_cells = extract_cell_count_from_files(dict_input_file_paths)
    dict_mouse_colors = {
        mouse_id: ddoc.get_color_for_mouse_id(mouse_id) for mouse_id in dict_n_cells
    }
    # get analysis type
    analysis_type = get_dataset_type(dict_n_cells)
    dict_n_frames = extract_frame_count_from_files(dict_input_file_paths)
    df_results = create_results(
        dict_n_cells=dict_n_cells,
        dict_mouse_colors=dict_mouse_colors,
        dict_n_frames=dict_n_frames,
        data_documentation=ddoc,
    ).reset_index(drop=True)
    if save_results:
        df_results.to_excel(
            os.path.join(
                output_folder, f"cell_count_pre_post_{analysis_type}_{dtime}.xlsx"
            ),
            index=False,
        )
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fpath",
        type=str,
        default=None,
        help="Path to the json file containing fpaths to all data to open",
    )
    parser.add_argument(
        "--fpath_out",
        type=str,
        default=None,
        help="Path where to save the result (if --save_data,\
             overrides .env OUTPUT_FOLDER variable)",
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="Save data to Excel file, default: false",
    )
    args = parser.parse_args()
    # assume dataset root is the folder where the json file is
    root_dir = os.path.split(args.fpath)[0]
    dict_input_files = load_pre_post_files(args.fpath, root_dir)
    main(
        dict_input_file_paths=dict_input_files,
        save_results=args.save_data,
        output_folder=args.fpath_out,
    )
