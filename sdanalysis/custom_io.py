"""
custom_io.py - A collection of functions for file input/output operations.
"""
import datetime as dt
from typing import Tuple, List
from tkinter import Tk  # use tkinter to open files
from tkinter.filedialog import askopenfilename, askdirectory
import os.path
import warnings
import numpy as np
import pims_nd2  # pip install pims_nd2


# TODO: open_dir opens dialog in foreground (in Jupyter), thanks to root.attributes("-topmost", True). Implement this
#  in other dialog callign functions!


def raise_above_all(window):
    """Helper function to raise a tkinter window above all other windows.
    Args:
        window (tkinter.Window): _description_
    """
    window.attributes("-topmost", 1)
    window.attributes("-topmost", 0)


def open_file(title: str = "Select file") -> str:
    """
    Opens a tkinter dialog to select a file. Returns the path of the file.
    :param title: The message to display in the open directory dialog.
    :return: the absolute path of the directory selected.
    """
    root = Tk()
    # dialog should open on top. Only works for Windows?
    root.attributes("-topmost", True)
    root.withdraw()  # keep root window from appearing
    return os.path.normpath(askopenfilename(title=title))


def open_dir(title: str = "Select data directory", ending_slash: bool = False) -> str:
    """
    Opens a tkinter dialog to select a folder. Returns the path of the folder.
    :param title: The message to display in the open directory dialog.
    :return: the absolute path of the directory selected.
    """
    root = Tk()
    # dialog should open on top. Only works for Windows?
    root.attributes("-topmost", True)
    root.withdraw()  # keep root window from appearing
    folder_path = askdirectory(title=title)
    if ending_slash:
        folder_path += "/"
    return os.path.normpath(folder_path)


def choose_dir_for_saving_file(
    title: str = "Select a folder to save the file to", fname: str = "output_file.txt"
):
    """
    Opens a tkinter dialog to select a folder. Returns opened folder + file name as path string.
    :param title:
    :param fname:
    :return:
    """
    return os.path.normpath(os.path.join(open_dir(title), fname))


def get_filename_with_date(raw_filename: str = "output_file", extension: str = ".txt"):
    """
    Given a root filename raw_filename, create an extended filename with extension. This avoids overwriting files saved
    repeatedly to the same folder by appending the date and time (including the seconds).
    :param raw_filename: file name without extension
    :param extension: the desired file extension. It should include the '.'!
    :return:
    """
    # todo: this should be a bit more sophisticated. (dealing with cases like extension without "." etc.), getting rid
    #  of extension in raw_filename if supplied...
    # dt.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    datetime_suffix = get_datetime_for_fname()
    return raw_filename + "_" + datetime_suffix + extension


def get_datetime_for_fname():
    """Get the current date and time in a format suitable for appending to a filename.

    Returns:
        str: The date and time as a string
    """
    now = dt.datetime.now()
    return f"{now.year:04d}{now.month:02d}{now.day:02d}-{now.hour:02d}{now.minute:02d}{now.second:02d}"


def np_arr_from_nd2(nd2_fpath: str, begin_end_frames: Tuple[int, int] = None):
    """Given an nd2 file, open it as a numpy array, then close it. If begin_end_frames is provided,
    only specific frames will be returned (the frames are 1-indexed):
    (1, 5) means frame #1 to frame #5 will be in the array.
    [(1,5), (7,8)] means frames #1 to #5, followed by #7 and #8 wiill be in the array.
    Args:
        nd2_fpath (str): _description_
        begin_end_frames (Tuple[int, int], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # set iter_axes to "t"
    # then: create nd array with sizes matching frame size,
    # begin_end_frames are 1-indexed, i.e. frame 1, 2, ...
    # begin_end_frames might be tuple (if single segment) or a list of tuples as sorted subsequent, non-overlapping segments.
    #
    with pims_nd2.ND2_Reader(nd2_fpath) as nikon_file:  # todo: get metadata too?
        sizes_dict = nikon_file.sizes
        if begin_end_frames is not None:
            if isinstance(begin_end_frames, tuple):
                sizes = (
                    begin_end_frames[1] - begin_end_frames[0] + 1,
                    sizes_dict["x"],
                    sizes_dict["y"],
                )
                # convert to list of tuple(s) for looping
                begin_end_frames = [begin_end_frames]
            elif isinstance(begin_end_frames, list):
                if len(begin_end_frames) == 0 or not (
                    isinstance(begin_end_frames[0], tuple)
                    or isinstance(begin_end_frames[0], list)
                ):
                    raise ValueError(
                        f"Expected begin_end_frames tuple of int, list of tuple or list of list, got: {type(begin_end_frames)}"
                    )
                n_frames = 0
                n_frames_nd2 = sizes_dict["t"]
                # check proper format of list begin_end_frames:
                # 1. should contain tuples or lists or 2 elements
                # 2. the entries should be int
                # 3. each pair should be either equal (segment of length 1) or ascending order. i.e. beginning frame <= end frame
                # 4. each segment should start after the previous, i.e. all segments are non-overlapping and subsequent
                # 5. any segments that start after the length of the passed recording are ignored (removed from the list)
                # 6. the last segment must end on or before the last frame of the nd2 file
                i_segment = 0
                for begin_end_frame in begin_end_frames:
                    if len(begin_end_frame) != 2:  # 1.
                        raise ValueError(
                            f"Expected begin_end_frames to contain lists or tuples of 2 ints, found {len(begin_end_frame)}: {begin_end_frame}"
                        )
                    # TODO: add proper working dtpye check that allows int and np.int*, maybe even np.uint*
                    # if not (np.issubdtype(type(begin_end_frame[0]), int) and np.issubdtype(type(begin_end_frame[1]), int)):  # 2.
                    #    raise ValueError(f"Expected begin_end_frames to contain lists or tuples of ints, found types {type(begin_end_frame[0])}, {type(begin_end_frame[1])}")
                    if begin_end_frame[0] > begin_end_frame[1]:  # 3.
                        raise ValueError(
                            f"begin_end_frames: begin frame greater than end frame: {begin_end_frame}"
                        )
                    if i_segment > 0:
                        prev_begin_end_frame = begin_end_frames[i_segment - 1]
                        # 4., previous segment already asserted to have begin <= end
                        if begin_end_frame[0] <= prev_begin_end_frame[1]:
                            raise ValueError(
                                f"begin_end_frames: overlapping segments defined: {prev_begin_end_frame}, {begin_end_frame}"
                            )
                    # 5. in 1-indexing, last index would be n_frames_nd2. Last possible segment starts with frame #n_frames_nd2
                    if begin_end_frame[0] > n_frames_nd2:
                        warnings.warn(
                            f"Segment out of recording length: {begin_end_frame} first and last frame (1-indexing), recording has {n_frames_nd2} frames"
                        )
                        break
                    if begin_end_frame[1] > n_frames_nd2:  # 6.
                        warnings.warn(
                            f"Segment cut to recording length: {begin_end_frame} cut to {(begin_end_frame[0], n_frames_nd2)}"
                        )
                        n_frames += n_frames_nd2 - begin_end_frame[0] + 1
                        begin_end_frames[i_segment] = (begin_end_frame[0], n_frames_nd2)
                        i_segment += 1  # this segment should still be included
                        break  # no future segments can be added, but avoid raising an accidental error in the checks for next segment
                        # (i.e. segment fails 1-4 results in error, even though the segment will not be included)
                    n_frames += begin_end_frame[1] - begin_end_frame[0] + 1
                    i_segment += 1
                begin_end_frames = begin_end_frames[:i_segment]  # 5.
                sizes = (n_frames, sizes_dict["x"], sizes_dict["y"])
        else:
            sizes = (sizes_dict["t"], sizes_dict["x"], sizes_dict["y"])
            # need list of tuple(s) for looping below
            begin_end_frames = [(1, sizes_dict["t"])]

        # dtype would be float32 by default...
        frames_arr = np.zeros(sizes, dtype=nikon_file.pixel_type)
        i_arr_frame = 0  # pointer of current frame in frames_arr to be written

        for begin_end_frame in begin_end_frames:
            # for debugging, make sure we are working with list of tuples or lists (begin, end frame indices)
            assert isinstance(begin_end_frame, tuple) or isinstance(
                begin_end_frame, list
            )
            # convert to 0-indexing
            for i_frame in range(begin_end_frame[0] - 1, begin_end_frame[1]):
                frames_arr[i_arr_frame] = np.array(
                    nikon_file[i_frame], dtype=nikon_file.pixel_type
                )
                i_arr_frame += 1
        return frames_arr


def np_arr_and_time_stamps_from_nd2(nd2_fpath: str, begin_end_frames: List[int] = None):
    """
    :param nd2_fpath: str
    :param begin_end_frames: (Optional) in 1-indexing, the indices of the first and last frames of the sequence to be
    imported
    :return:
    frames_arr: 3D array of dims (T, X, Y)
    tstamps_arr: 1x(length of segment) np.float64 array of time stamps in float (ms)
    """
    # set iter_axes to "t"
    # then: create nd array with sizes matching frame size,
    with pims_nd2.ND2_Reader(nd2_fpath) as nikon_file:  # todo: get metadata too?
        # get begin and end frames in 0-indexing
        sizes_dict = nikon_file.sizes
        if begin_end_frames is not None:
            i_begin = begin_end_frames[0] - 1
            i_end = begin_end_frames[1] - 1
        else:
            print(
                f"Begin and end frames not provided; reading whole video ({sizes_dict['t']})"
            )
            i_begin = 0
            i_end = nikon_file.sizes["t"] - 1
        sizes = (i_end - i_begin + 1, sizes_dict["x"], sizes_dict["y"])

        # dtype would be float32 by default...
        frames_arr = np.zeros(sizes, dtype=nikon_file.pixel_type)
        tstamps_arr = np.zeros(sizes[0], dtype=np.float64)
        # TODO: probably it is not even necessary to export an np.array, as nikon_file is an iterable of
        #  subclasses of np array... not sure what caiman needs
        # frames_arr should be filled from 0, but the segment might not start from frame 0
        i_arr_element = 0
        for i_frame in range(i_begin, i_end + 1):
            # not sure if dtype needed here
            frames_arr[i_arr_element] = np.array(
                nikon_file[i_frame], dtype=nikon_file.pixel_type
            )
            tstamps_arr[i_arr_element] = nikon_file[i_frame].metadata["t_ms"]
            i_arr_element += 1
        return frames_arr, tstamps_arr
