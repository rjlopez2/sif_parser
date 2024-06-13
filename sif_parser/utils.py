import typing
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import sif_open as sif


def extract_calibration(info):
    """
    Extract calibration data from info.

    Parameters
    ----------
    info: OrderedDict
        OrderedDict from np_open

    Returns
    -------
    calibration:
        np.ndarray.
        1d array sized [width] if only 1 calibration is found.
        2d array sized [NumberOfFrames x width] if multiple calibration is
            found.
        None if no calibration is found
    """
    width = info['DetectorDimensions'][0]
    # multiple calibration data is stored
    if 'Calibration_data_for_frame_1' in info:
        calibration = np.ndarray((info['NumberOfFrames'], width))
        for f in range(len(calibration)):
            key = 'Calibration_data_for_frame_{:d}'.format(f + 1)
            flip_coef = np.flipud(info[key])
            calibration[f] = np.poly1d(flip_coef)(np.arange(1, width + 1))
        return calibration

    elif 'Calibration_data' in info:
        flip_coef = np.flipud(info['Calibration_data'])
        return np.poly1d(flip_coef)(np.arange(1, width + 1))
    else:
        return None


def parse(file: str) -> typing.Tuple[np.ndarray, typing.Dict]:
    """
    Parse a .sif file.

    :param file: Path to a `.sif` file.
    :returns tuple[numpy.ndarray, OrderedDict]: Tuple of (data, info) where
        `data` is an (channels x 2) array with the first element of each row
        being the wavelength bin and the second being the counts.
        `info` is an OrderedDict of information about the measurement.
    """
    data, info = sif.np_open(file)
    wavelengths = extract_calibration(info)

    # @todo: `data.flatten()` may not be compatible with
    #   multiple images or 2D images.
    df = np.column_stack((wavelengths, data.flatten()))
    return (df, info)

def ordered_dat_files(input_string):
    """
    This helper function sort the list of .dat files in the exspected order
    """
    # Step 1: Remove non-numeric characters
    numeric_string = ''.join(filter(str.isdigit, os.path.basename(input_string)))

    # Step 2: Reverse the string
    reversed_string = numeric_string[::-1]

    # Step 3: Convert the reversed string to an integer
    result = int(reversed_string)

    # Print the result
    return result



def read_dat_image(filepath, y_, x_, datatype):
    data = np.fromfile(filepath, offset=0, dtype=datatype, count=y_ * x_).reshape(y_, x_)
    return data


def image_generator(dat_files_list):
    for filepath in dat_files_list:
        yield filepath


def read_dat_images_in_multithread(x_, y_, dat_files_list, datatype, pre_alocated_array, max_workers=8):
    t = len(dat_files_list)
    #pre_alocated_array = np.empty([t, y_, x_], datatype)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_frame = {executor.submit(read_dat_image, dat_files_list[frame], y_, x_, datatype): frame for frame in range(t)}
        
        for future in as_completed(future_to_frame):
            frame = future_to_frame[future]
            try:
                pre_alocated_array[frame, ...] = future.result()
            except Exception as exc:
                print(f'Frame {frame} generated an exception: {exc}')
    
    return pre_alocated_array


def read_dat_images_in_multithread_with_generator(x_, y_, dat_files_list, datatype, pre_alocated_array, max_workers=8):
    image_paths = list(image_generator(dat_files_list))
    t = len(image_paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_frame = {executor.submit(read_dat_image, image_paths[frame], y_, x_, datatype): frame for frame in range(t)}
        
        for future in as_completed(future_to_frame):
            frame = future_to_frame[future]
            try:
                pre_alocated_array[frame, ...] = future.result()
            except Exception as exc:
                print(f'Frame {frame} generated an exception: {exc}')
    
    return pre_alocated_array
