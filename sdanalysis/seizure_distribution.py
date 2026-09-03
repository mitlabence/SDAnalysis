import h5py
from typing import Tuple, Dict
import numpy as np
import pandas as pd

def pad_array(arr, n_events, pad_value):
    pad_length = n_events - len(arr)
    if pad_length > 0:
        arr = np.concatenate((arr, np.full(pad_length, pad_value)))
    return arr


def open_sz_distribution_file(fpath_sz_data: str) -> Tuple[Dict[str, np.ndarray], int, int]:
    """
    Opens an HDF5 file containing seizure data and returns a dictionary with mouse IDs as keys and their seizure times as values.
    """
    sz_times_dict = {}
    n_max_sz = 0
    with h5py.File(fpath_sz_data, "r") as hf:
        for mouse_id in hf.keys():
            all_sz = hf[mouse_id]["all_seizures"][()]
            sz_times_dict[mouse_id] = all_sz
            n_max_sz = max(n_max_sz, len(all_sz))
    # filter out mice with no seizures and pad the arrays to have the same length
    sz_times_dict_filtered = dict()
    for mouse_id in sz_times_dict.keys():
        if len(sz_times_dict[mouse_id]) > 0:
            sz_times_dict_filtered[mouse_id] = pad_array(sz_times_dict[mouse_id], n_max_sz, np.nan)
    n_mice = len(sz_times_dict_filtered.keys())

    return sz_times_dict_filtered, n_max_sz, n_mice

def sz_distribution_df_short(sz_times_dict_filtered: Dict[str, np.ndarray], n_max_sz: int, n_mice: int) -> pd.DataFrame:
    """
    Converts the seizure times dictionary to a pandas DataFrame, where each column represents a mouse and each row represents a seizure time.

    Args:
        sz_times_dict_filtered (Dict[str, np.ndarray]): Dictionary with mouse IDs as keys and their seizure times as values.
        n_max_sz (int): The maximum number of seizures across all mice.
        n_mice (int): The number of mice with at least one seizure.
    
    Returns:
        pd.DataFrame: DataFrame with each column representing a mouse, column names the mouse IDs, sorted by number of seizures (descending),
        and the entries in the rows representing the (ascending) time (in hours) of a seizure after injection.
    """
    seizure_times_array = np.zeros((n_max_sz, n_mice))  # each column is a mouse
    mice = list(sz_times_dict_filtered.keys())
    for i, mouse_id in enumerate(mice):
        seizure_times_array[:, i] = sz_times_dict_filtered[mouse_id]
    # short format df: columns are mice, rows are seizure times, NaN for missing values to have same length columns
    df_sz_times_short = pd.DataFrame(data=seizure_times_array, columns = mice)
    # sort df columns by number of Sz descending (= NaN ascending), then by mouse ID (if same number of Sz)
    nan_counts = df_sz_times_short.isna().sum()
    sorted_columns = nan_counts.sort_values(ascending=True).index
    df_sorted = df_sz_times_short[sorted_columns]
    return df_sorted

def sz_distribution_df_long(sz_times_dict_filtered, n_max_sz, n_mice) -> pd.DataFrame:
    """
    Converts the seizure times dictionary to a long format pandas DataFrame, where each row represents a seizure event with mouse ID and seizure time.

    Args:
        sz_times_dict_filtered (Dict[str, np.ndarray]): Dictionary with mouse IDs as keys and their seizure times as values.
        n_max_sz (int): The maximum number of seizures across all mice.
        n_mice (int): The number of mice with at least one seizure.
    
    Returns:
        pd.DataFrame: long format DataFrame with two columns: 'mouse' and 'seizure_time', where each row represents a seizure event.
    """
    df_sz_times_short = sz_distribution_df_short(sz_times_dict_filtered, n_max_sz, n_mice)
    return sz_distribution_df_short_to_long(df_sz_times_short)


def sz_distribution_df_short_to_long(df_sz_times_short: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the short format seizure times DataFrame to a long format DataFrame, where each row represents a seizure event with mouse ID and seizure time.

    Args:
        df_sz_times_short (pd.DataFrame): DataFrame with each column representing a mouse, column names the mouse IDs, sorted by number of seizures (descending),
        and the entries in the rows representing the (ascending) time (in hours) of a seizure after injection.
    Returns:
        pd.DataFrame: long format DataFrame with two columns: 'mouse' and 'seizure_time', where each row represents a seizure event.
    """
    # make into long format by melting the dataframe (two columns: mouse ID and seizure time)
    return pd.melt(df_sz_times_short, var_name='mouse', value_name='seizure_time').dropna().sort_values(["mouse", "seizure_time"]).reset_index(drop=True)
