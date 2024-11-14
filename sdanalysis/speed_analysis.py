"""
speed_analysis.py - Calculate the speed of the wavefronts in the directionality data, 
shares methods with directionality_analysis.py
"""
import os
from collections.abc import Iterable
import argparse

# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from sdanalysis.custom_io import open_dir, get_datetime_for_fname
import sdanalysis.data_documentation

# multiprocessing does not work with IPython. Use fork instead.

from sdanalysis.env_reader import read_env
from sdanalysis.directionality_analysis import (
    get_directionality_files_list,
    directionality_files_to_df,
    create_seizure_uuid,
    replace_multiple_outliers,
)


def speeds_per_session(
    df_onsets_input,
    i_wave,
    n_neighbors=1,
    vectorize=False,
    onset_sz=False,
):
    """_summary_

    Parameters
    ----------
    df_onsets_input : _type_
        _description_
    i_wave : int
        index of wave, should be 1 or 2
    n_neighbors : int, optional
        average the closest n_neighbors cells (with a later onset), by default 1
    vectorize : bool, optional
         whether to return not only the velocity, but in addition, the 2d vector velocity,
         as well as the centre of the neuron, by default False
    onset_sz : bool, optional
        _description_, by default False

    Returns
    -------
    tuple
        uuids: a list of the uuids, and a 2d list of velocities: an array of all calculated
        velocities per session (uuid_extended)
    """
    uuids = []
    vs_2d = []
    neuron_ids = np.array([], dtype=np.int16)
    if onset_sz:
        onset_type = "onset_sz"
    else:
        onset_type = "onset" + str(i_wave)

    if vectorize:
        dx_2d = []
        dy_2d = []
        # the centre coordinate of each neuron. Same as "x" column in all_onsets_df.
        centres_x = np.array([])
        centres_y = np.array([])
    for i_group, session_group in df_onsets_input[
        df_onsets_input[onset_type].notna()
    ].groupby("uuid_extended"):
        # TODO: the center values should be mean, not median!
        x_y_onset = np.array(
            [session_group["x"], session_group["y"], session_group[onset_type]]
        )
        x_y_onset = x_y_onset.T  # x_y_onset1[i] = [x_i, y_i, onset1_i]
        n_neurons = len(x_y_onset)
        neuron_ids_curr_session = np.array(session_group["neuron_id"], dtype=np.int16)
        neuron_ids = np.concatenate([neuron_ids, neuron_ids_curr_session])
        # contains (mean) x/y distance to nearest neighbor for each neuron

        # 1. find all neurons with later onset
        #      boolean array of arrays: in a row i, value at index j is True if onset j
        # is greater than onset i.
        larger_values = x_y_onset[:, 2][:, np.newaxis] < x_y_onset[:, 2]
        #      convert True/False into index. Use fact that within a row, i-th element
        # corresponds to index i.
        # Put np.inf if not larger
        larger_indices = np.where(larger_values, np.arange(n_neurons), np.inf)
        # 2. find all neuron distances
        dist_matrix = distance_matrix(x_y_onset[:, :2], x_y_onset[:, :2])
        # dist_matrix: each row contains distance to all the other tiles. inf if same tile!
        # (diagonal)
        assert (dist_matrix == dist_matrix.T).all()  # symmetric
        # exclude tile itself from being nearest neighbor
        np.fill_diagonal(dist_matrix, np.inf)
        # find distances neurons with later onset
        later_neurons_distances = np.where(
            np.isfinite(larger_indices), dist_matrix, np.inf
        )
        # find closest neurons with later onset
        nearest_indices_later_onset = np.argsort(later_neurons_distances, axis=1)[
            :, :n_neighbors
        ]
        # calculate velocity with all neighbors above
        vs = np.zeros(n_neurons)
        if vectorize:
            dxs = np.zeros(n_neurons)
            dys = np.zeros(n_neurons)
        for i_neuron, neuron_nearest_indices in enumerate(nearest_indices_later_onset):
            # a later onset neuron is actually found
            if np.isinf(later_neurons_distances[i_neuron]).all():
                continue
            else:
                if isinstance(neuron_nearest_indices, Iterable):
                    v_neighbors_list = np.zeros(len(neuron_nearest_indices))
                    if vectorize:
                        dx_neighbors_list = np.zeros(len(neuron_nearest_indices))
                        dy_neighbors_list = np.zeros(len(neuron_nearest_indices))

                    for i_neighbor, index_neighbor in enumerate(neuron_nearest_indices):
                        # objective conversion factor  -> [pixel] * [µm] / [pixel]
                        ds = dist_matrix[i_neuron][index_neighbor] * 1.579
                        # [frames] / ([frames]/[second])
                        dt = (
                            x_y_onset[index_neighbor][2] - x_y_onset[i_neuron][2]
                        ) / 15.0
                        v_neighbor = ds / dt
                        v_neighbors_list[i_neighbor] = v_neighbor
                        if vectorize:  #
                            # get x, y of current neighbor
                            x_nearest = x_y_onset[index_neighbor][0]
                            y_nearest = x_y_onset[index_neighbor][1]
                            # get x, y of current neuron
                            x_curr = x_y_onset[i_neuron][0]
                            y_curr = x_y_onset[i_neuron][1]
                            # get dx, dy
                            dx = x_nearest - x_curr
                            dy = y_nearest - y_curr
                            dx_neighbors_list[i_neighbor] = dx
                            dy_neighbors_list[i_neighbor] = dy

                    vs[i_neuron] = np.median(v_neighbors_list)
                    if vectorize:
                        dxs[i_neuron] = np.median(dx_neighbors_list)
                        dys[i_neuron] = np.median(dy_neighbors_list)

                else:
                    # objective conversion factor  -> [pixel] * [µm] / [pixel]
                    ds = dist_matrix[i_neuron][neuron_nearest_indices[0]] * 1.579
                    # [frames] / ([frames]/[second])
                    dt = (
                        x_y_onset[neuron_nearest_indices[0]][2] - x_y_onset[i_neuron][2]
                    ) / 15.0
                    vs[i_neuron] = ds / dt
        vs_2d.append(vs)
        uuids.append(i_group)
        if vectorize:
            centres_x = np.concatenate([centres_x, session_group["x"]])
            centres_y = np.concatenate([centres_y, session_group["y"]])
            dx_2d.append(dxs)
            dy_2d.append(dys)

    vs_flat = [item for vs_row in vs_2d for item in vs_row]
    v_median = np.median(vs_flat)
    print(f"{v_median} µm/s = {v_median*6./100.} mm/min")
    if vectorize:
        return (uuids, neuron_ids, vs_2d, dx_2d, dy_2d, centres_x, centres_y)
    return (uuids, neuron_ids, vs_2d, None, None, None, None)  # in µm/s


def extended_to_normal_uuid(extended_uuid: str) -> str:
    """Given a uuid in the extended format (either <uuid> or <uuid>_1, <uuid>_2, etc, unique for
    each seizure in the recording with <uuid>), get the original recording uuid (<uuid>)

    Parameters
    ----------
    extended_uuid : str
        the exended uuid of an event

    Returns
    -------
    str
        the uuid of the corresponding recording
    """
    if "_" in extended_uuid:
        return extended_uuid.split("_")[0]
    else:
        return extended_uuid


def main(
    folder: str,
    save_data: bool,
    # save_figs: bool = False,
    # file_format: str = "pdf",
    n_neighbors: int = 1,
):
    # TODO: option to choose output file format: excel (xlsx) vs hdf5
    # get datetime for output file name
    output_dtime = get_datetime_for_fname()
    replace_outliers = True  # TODO: add it as a command line argument
    env_dict = read_env()
    data_doc = sdanalysis.data_documentation.DataDocumentation.from_env_dict(env_dict)
    if save_data:
        output_folder = env_dict["OUTPUT_FOLDER"]
    else:
        output_folder = None
    if folder is None or not os.path.exists(folder):
        folder = open_dir("Open directory with directionality data")
    analysis_fpaths = get_directionality_files_list(folder)
    df_onsets = directionality_files_to_df(analysis_fpaths, data_doc)
    dict_uuid_exp_type = {
        uuid: data_doc.get_experiment_type_for_uuid(uuid)
        for uuid in df_onsets.uuid.unique()
    }
    # for old files, "i_sz" is not a column, as only one seizure per recording was found.
    # Add this column here
    if "i_sz" not in df_onsets.columns:
        df_onsets["i_sz"] = np.nan
    # make a uuid unique to seizure, call the column "uuid_extended"
    df_onsets["uuid_extended"] = df_onsets.apply(create_seizure_uuid, axis=1)
    if replace_outliers:
        df_onsets = replace_multiple_outliers(
            df_onsets, ["onset1", "onset2", "onset_sz"], percent=0.05
        )
    uuids_neuron1, _, vs_neuron1, _, _, _, _ = speeds_per_session(
        df_onsets, 1, n_neighbors
    )
    uuids_neuron2, _, vs_neuron2, _, _, _, _ = speeds_per_session(
        df_onsets, 2, n_neighbors
    )
    uuids_neuron_sz, _, vs_neuron_sz, _, _, _, _ = speeds_per_session(
        df_onsets, 2, n_neighbors, False, True
    )
    # flatten all arrays
    vs_neuron_sz_flat = [element for sublist in vs_neuron_sz for element in sublist]
    uuids_neuron_sz_flat = [
        uuids_neuron_sz[i]
        for i, neurons in enumerate(vs_neuron_sz)
        for j in range(len(neurons))
    ]
    vs_neuron1_flat = [element for sublist in vs_neuron1 for element in sublist]
    uuids_neuron1_flat = [
        uuids_neuron1[i]
        for i, neurons in enumerate(vs_neuron1)
        for j in range(len(neurons))
    ]
    vs_neuron2_flat = [element for sublist in vs_neuron2 for element in sublist]
    uuids_neuron2_flat = [
        uuids_neuron2[i]
        for i, neurons in enumerate(vs_neuron2)
        for j in range(len(neurons))
    ]
    vs_neuron_sz_mean = [np.median(element) for element in vs_neuron_sz]
    assert len(vs_neuron_sz_flat) == len(uuids_neuron_sz_flat)
    assert len(vs_neuron1_flat) == len(uuids_neuron1_flat)
    assert len(vs_neuron2_flat) == len(uuids_neuron2_flat)
    # neuron-based algorithm
    # all velocities calculated
    vs_df1 = pd.DataFrame(
        {"uuid": uuids_neuron1_flat, "v_umps": vs_neuron1_flat, "i_wave": 1}
    )
    # all velocities calculated
    vs_df2 = pd.DataFrame(
        {"uuid": uuids_neuron2_flat, "v_umps": vs_neuron2_flat, "i_wave": 2}
    )
    # all velocities calculated
    vs_df_sz = pd.DataFrame(
        {"uuid": uuids_neuron_sz_flat, "v_umps": vs_neuron_sz_flat, "i_wave": 0}
    )
    vs_df_sz_means = pd.DataFrame(
        {"uuid": uuids_neuron_sz, "v_umps": vs_neuron_sz_mean, "i_wave": 1}
    )

    # reset index, but keep old index just in case
    vs_df = pd.concat([vs_df1, vs_df2], axis=0).reset_index()
    vs_df_sz = vs_df_sz.reset_index()

    # get rid of 0 values
    vs_df = vs_df[vs_df["v_umps"] > 0.0]
    vs_df_sz = vs_df_sz[vs_df_sz["v_umps"] > 0.0]

    vs_df["mouse_id"] = vs_df.apply(
        lambda row: data_doc.get_mouse_id_for_uuid(extended_to_normal_uuid(row["uuid"])),
        axis=1,
    )
    vs_df_sz["mouse_id"] = vs_df_sz.apply(
        lambda row: data_doc.get_mouse_id_for_uuid(extended_to_normal_uuid(row["uuid"])),
        axis=1,
    )

    vs_df["exp_type"] = vs_df.apply(
        lambda row: dict_uuid_exp_type[extended_to_normal_uuid(row["uuid"])], axis=1
    )
    vs_df_sz["exp_type"] = vs_df_sz.apply(
        lambda row: dict_uuid_exp_type[extended_to_normal_uuid(row["uuid"])], axis=1
    )
    vs_df_sz_means["exp_type"] = vs_df_sz_means.apply(
        lambda row: dict_uuid_exp_type[extended_to_normal_uuid(row["uuid"])], axis=1
    )

    # convert to mm/min
    conversion_factor = 0.06  # 1 um/s = 60 um/min = 0.06 mm/min
    vs_df["v_mmpmin"] = vs_df["v_umps"] * conversion_factor
    vs_df_sz["v_mmpmin"] = vs_df_sz["v_umps"] * conversion_factor

    # create dataset with outliers removed
    vs_df_sz_outliers_removed = vs_df_sz.copy()
    for _, g in vs_df_sz_outliers_removed.groupby("uuid"):
        count = g.size
        drop = int(count * 0.05)  # drop lowest and highest 5%
        vs_df_sz_outliers_removed.drop(g["v_mmpmin"].nlargest(drop).index, inplace=True)
        vs_df_sz_outliers_removed.drop(
            g["v_mmpmin"].nsmallest(drop).index, inplace=True
        )
    means_per_session = (
        vs_df.groupby(["exp_type", "mouse_id", "uuid", "i_wave"]).median().reset_index()
    )  # mean().reset_index()
    means_per_session["speed_type"] = "SD"
    sz_means_per_session = (
        vs_df_sz_outliers_removed.groupby(["exp_type", "mouse_id", "uuid"])
        .median()
        .reset_index()
    )  # vs_df_sz.groupby(["exp_type", "mouse_id", "uuid"]).mean().reset_index()
    sz_means_per_session["speed_type"] = "Sz"
    df_mean_speeds = pd.concat(
        [
            means_per_session[
                [
                    "exp_type",
                    "mouse_id",
                    "uuid",
                    "speed_type",
                    "i_wave",
                    "v_umps",
                    "v_mmpmin",
                ]
            ],
            sz_means_per_session[
                [
                    "exp_type",
                    "mouse_id",
                    "uuid",
                    "speed_type",
                    "i_wave",
                    "v_umps",
                    "v_mmpmin",
                ]
            ],
        ]
    ).reset_index(drop=True)
    if save_data:
        export_fpath_df_mean_speeds = os.path.join(
            output_folder, f"mean_onset_speed_{output_dtime}.xlsx"
        )
        df_mean_speeds.to_excel(export_fpath_df_mean_speeds, index=False)
    return (df_mean_speeds,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Folder with directionality data h5 files",
    )
    parser.add_argument(
        "--save_data", action="store_true", help="Save data to Excel file"
    )
    parser.add_argument("--save_figs", action="store_true", help="Save figures")
    parser.add_argument(
        "--file_format", type=str, default="pdf", help="File format for figures"
    )
    parser.add_argument("--n_neighbors", type=int, default=1)
    args = parser.parse_args()
    main(
        args.folder, args.save_data, args.save_figs, args.file_format, args.n_neighbors
    )
