"""
cell_count_analysis.py - This script is used to analyze the number of cells in each recording of a
specified dataset.
"""

import os
import json
import argparse
from typing import Tuple
import h5py
import pandas as pd
import data_documentation as dd
import custom_io as cio
import env_reader


def create_results(dict_n_cells: dict, dict_mouse_colors: dict) -> pd.DataFrame:
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
        dict_mouse_colors (dict): _description_

        dict_n_cells[list(dict_n_cells.keys())[0]]

    Returns:
        pd.DataFrame: _description_
    """
    col_mouse = []  # mouse id
    col_exp_type = []  # szsd, sd, ctl
    col_uuid = []  # unique for recording
    col_cell_count_pre = []
    col_cell_count_post = []
    col_colors = []

    for mouse_id, exp_types_for_mouse in dict_n_cells.items():
        for exp_type, n_cells_pre_post in exp_types_for_mouse.items():
            color = dict_mouse_colors[mouse_id]
            n_cells_pre = n_cells_pre_post["pre"]
            n_cells_post = n_cells_pre_post["post"]
            uuids = n_cells_pre_post["uuid"]
            mouse_ids = [mouse_id] * len(n_cells_pre)
            exp_types = [exp_type] * len(n_cells_pre)
            colors = [color] * len(n_cells_pre)

            col_mouse.extend(mouse_ids)
            col_exp_type.extend(exp_types)
            col_uuid.extend(uuids)
            col_cell_count_pre.extend(n_cells_pre)
            col_cell_count_post.extend(n_cells_post)
            col_colors.extend(colors)

    df_results = pd.DataFrame(
        {
            "mouse_id": col_mouse,
            "exp_type": col_exp_type,
            "uuid": col_uuid,
            "cell_count_pre": col_cell_count_pre,
            "cell_count_post": col_cell_count_post,
            "color": col_colors,
        }
    )
    df_results = df_results.sort_values(by=["mouse_id", "exp_type", "uuid"])
    df_results["post_pre_ratio"] = (
        df_results["cell_count_post"] / df_results["cell_count_pre"]
    )
    return df_results


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


def extract_data_from_files(dict_fpaths: dict) -> dict:
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
    # {mouse_id: color}
    dict_mouse_colors = {}
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
                    n_cells_pre.append(hdf_file["estimates"]["A"]["shape"][1])
                    uuids.append(hdf_file.attrs["uuid"])
                with h5py.File(fpath_post, "r") as hdf_file:
                    n_cells_post.append(hdf_file["estimates"]["A"]["shape"][1])
            dict_n_cells[mouse_id][exp_type]["pre"] = n_cells_pre
            dict_n_cells[mouse_id][exp_type]["post"] = n_cells_post
            dict_n_cells[mouse_id][exp_type]["uuid"] = uuids
    return dict_n_cells


def load_pre_post_files(fpath_input: str) -> dict:
    """
    Load the json file containing the file paths of the data to open. It is expected to
    have the form:
    {mouse_id:
        {exp_type:
            {"pre": [fpaths for each recording pre stage],
            "post": [fpaths for each recording post stage]}
        }
    }

    Args:
        fpath_input (str): The absolute file path of the json file

    Returns:
        dict: The dictionary containing the file paths in same format as the input json file
    """
    with open(fpath_input, "r", encoding="utf-8") as json_file:
        dict_fpaths = json.load(json_file)
    return dict_fpaths


def main(
    fpath_input: str = None, save_results: bool = False, output_folder: str = None
) -> pd.DataFrame:
    """_summary_

    Args:
        fpath_input (str): The file path of the json file that contains file paths of all data to
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
    if not (
        fpath_input is not None
        and os.path.splitext(fpath_input)[1] == ".json"
        and os.path.exists(fpath_input)
    ):
        raise ValueError("Invalid input file path")
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
    dict_fpaths = load_pre_post_files(fpath_input)

    with open(fpath_input, "r", encoding="utf-8") as json_file:
        dict_fpaths = json.load(json_file)
    dict_n_cells = extract_data_from_files(dict_fpaths)
    dict_mouse_colors = {
        mouse_id: ddoc.get_color_for_mouse_id(mouse_id) for mouse_id in dict_n_cells
    }
    # get analysis type
    analysis_type = get_dataset_type(dict_n_cells)
    df_results = create_results(
        dict_n_cells=dict_n_cells, dict_mouse_colors=dict_mouse_colors
    )
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
    main(
        fpath_input=args.fpath,
        save_results=args.save_data,
        output_folder=args.fpath_out,
    )
