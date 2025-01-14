# the parameters from the LabView matching & processing pipeline
# TODO add missing ones
PARAMS = [
    "path_name",
    "belt_file_name",
    "nikon_file_name",
    "tstamps_file_name",
    "lvstamps_len",
    "missed_frames",
    "missed_cycles",
    "i_belt_start",
    "i_belt_stop",
    "used_tstamps",
    "len_tsscn",
    "timestamps_were_duplicate",
    "len_tsscn_new",
    "movie_length_min",
    "frequency_estimated",
    "ard_thresneg",
    "ard_threspos",
    "art_n_artifacts",
    "belt_length_mm",
    "belt_input_thres",
    "belt_interrunning_window",
]


# the python variable names are more descriptive; for compatibility, the old Matlab variable names
# are kept in some places. Use this mapping to convert between the two. It contains all the relevant
# variables, but not all of what the data contains.

DICT_MATLAB_PYTHON_VARIABLES = {
    "speed": "speed",
    "distance": "total_distance",
    "round": "round",
    "distancePR": "distance_per_round",
    "reflect": "reflectivity",
    "stripes": "stripes_total",
    "stripesPR": "stripes_per_round",
    "time": "time_total_ms",
    "timePR": "time_per_round",
    "running": "running",
    "totdist_abs": "totdist_abs",
    "dt": "dt",
}  # the matlab original parameter names and their corresponding python naming

DICT_MATLAB_PYTHON_SCN_VARIABLES = {
    "speed": "speed",
    "distance": "distance_per_round",
    "rounds": "round",
    "totdist": "total_distance",
    "tsscn": "time_total_ms",
    "running": "running",
    "totdist_abs": "totdist_abs",
    "dt": "dt",
}
