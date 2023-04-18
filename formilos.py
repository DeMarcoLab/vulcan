import logging
import numpy as np

def convert_arr_to_streamfile(profile: np.ndarray,
                              protocol_settings: dict):

    save_path = protocol_settings.get("save_path")
    if not protocol_settings.get("create"):
        return f"{save_path}.str"

    #  read in protocol settings
    pixel_size = protocol_settings.get("pattern_pixel_size")
    pattern_width = profile.shape[0] * pixel_size
    pattern_height = profile.shape[1] * pixel_size

    # basic checks on pattern
    if not pattern_width <= protocol_settings.get("max_hfw")/protocol_settings.get("pattern_pixel_size"):
        logging.error(f"Pattern width must be less than or equal to {protocol_settings.get('max_hfw')}")
        raise ValueError(f"Pattern width must be less than or equal to {protocol_settings.get('max_hfw')}")
    if not pattern_height <= protocol_settings.get("max_hfw")/protocol_settings.get("pattern_pixel_size")*5/8: # aspect ratio of FIB
        logging.error(f"Pattern height must be less than or equal to {protocol_settings.get('max_hfw')*5/8}")
        raise ValueError(f"Pattern height must be less than or equal to {protocol_settings.get('max_hfw')*5/8}")

    if protocol_settings.get("invert"):
        profile = profile.max() - profile
        logging.info(f"Profile inverted")

    if protocol_settings.get("transpose"):
        profile = profile.T
        logging.info(f"Profile transposed")

    x, y = create_position_grid(profile=profile, protocol_settings=protocol_settings)
    logging.info(f"Position grid created successfully")

    # to cut through the top layer of the chip
    profile += protocol_settings.get("precut")

    percentage_profile = calculate_percentage_profile(profile)

    # calculate parameters
    n_doses = calculate_n_doses(profile=profile, protocol_settings=protocol_settings)

    if protocol_settings.get("scale"):
        # scale factor accounts for differences in mill time on UI against calculation, for a given current
        scale_factor = protocol_settings.get("scale_factor")
        if scale_factor is not None:
            logging.info(f"Using scale factor of {scale_factor}")
            n_doses = int(n_doses / scale_factor)
            logging.info(f"Scaled doses: {n_doses:e}")

    # calculate n_doses at each point of the profile
    dose_profile = percentage_profile * n_doses

    # use the minimum dose to calculate the number of passes
    minimum_doses = dose_profile.min()
    logging.info(f'Minimum n doses: {int(minimum_doses)}')
    maximum_doses = dose_profile.max()
    logging.info(f'Maximum n doses: {int(maximum_doses)}')
    
    # n passes is calculated so as to not over-cut any section
    n_passes = int(my_floor(dose_profile.min(), -2)/100)
    logging.info(f'Calculated number of passes: {n_passes}')

    n_passes = protocol_settings.get("n_passes")
    dt_profile = dose_profile/n_passes

    logging.info(f'Min doses post pass division: {dt_profile.min()}')
    logging.info(f'Max doses post pass division: {dt_profile.max()}')

    round_profile = my_floor(dt_profile, -2)/100
    logging.info(f'Total number of doses: {int(round_profile.sum())}')
    logging.info(f'Max doses post rounding: {int(round_profile.max())}')

    stream_list = []
    for i in range(dt_profile.shape[0]):
        for j in range(dt_profile.shape[1]):
            for point in range(int(round_profile[i, j])):
                
                stream_list.append([f'{point} 100', int(np.round(x[i, j])), int(np.round(y[i, j]))])
            # stream_list.append([int(np.round(dt_profile[i, j]/scale_factor)), int(np.round(x[i, j])), int(np.round(y[i, j]))])
            # stream_list.append([int(dt_profile[i, j]), int(np.round(x[i, j])), int(np.round(y[i, j]))])
    
    # sort the stream_list by the first value in each list
    logging.info(f"Stream list created successfully")

    if protocol_settings.get("shuffle"):
        np.random.shuffle(stream_list)
        logging.info(f"Stream list shuffled")
    stream_list.sort(key=lambda x: int(x[0].split(" ")[0]))

    # n_passes = protocol_settings.get("n_passes")


    with open(f"{save_path}.str", "w") as f:
        f.write("s16\n")
        f.write(str(int(n_passes))+"\n")
        f.write(str(int(round_profile.sum()))+"\n")
        for position in stream_list:
            f.write(f"{position[0].split(' ')[-1]} {position[1]} {position[2]}\n")

        logging.info(f"Stream file saved to {save_path}.str")

    return f"{save_path}.str"


def create_position_grid(profile: np.ndarray, protocol_settings: dict):
    """creates a grid of x and y coordinates for the milling pattern that matches the used pixel size, rather than the patterns pixel_size

    Parameters
    ----------
    profile : np.ndarray
        profile to be milled
    protocol_settings : dict
        protocol settings dictionary defining the pixel size of the pattern and the pixel size of the milling

    Returns
    -------
    x, y : np.ndarray
        mesh grid of x and y coordinates for the milling pattern in pixels
    """
    pattern_pixel_size = protocol_settings.get("pattern_pixel_size")
    pixel_size = protocol_settings.get("pixel_size")

    # create a grid of the same size as the profile with an extra dimension
    # the first dimension is the x coordinate
    # the second dimension is the y coordinate

    # calculate the number of pixels in the x and y direction
    x_pixels = profile.shape[1]
    y_pixels = profile.shape[0]

    # create the grid
    x = np.linspace(0, pattern_pixel_size*(x_pixels-1), x_pixels)
    y = np.linspace(0, pattern_pixel_size*(y_pixels-1), y_pixels)

    # create the meshgrid
    x, y = np.meshgrid(x, y)

    #TODO: magic number
    centre_x, centre_y = 2**16/2, 2**16/2

    position_x = protocol_settings.get("position_x")
    position_y = protocol_settings.get("position_y")

    position_x = np.round(position_x/pixel_size)
    position_y = np.round(position_y/pixel_size)

    centre_x_adjustment = -np.round((pattern_pixel_size*(x_pixels-1)/2)/pixel_size)
    centre_y_adjustment = -np.round((pattern_pixel_size*(y_pixels-1)/2)/pixel_size)

    # divide the meshgrid by the pixel_size
    x = np.round(x / pixel_size) + centre_x + position_x + centre_x_adjustment
    y = np.round(y / pixel_size) + centre_y + position_y + centre_y_adjustment

    return x, y


import numpy as np
from PIL import Image
import yaml
from pathlib import Path
import os

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from enum import Enum
import tifffile as tf

class ChipLocation(Enum):
    Centre = 1
    TopLeft = 2
    TopRight = 3
    BottomLeft = 4
    BottomRight = 5


def load_profile(path: Path) -> np.ndarray:

    return np.load(path)


def transform_profile(arr:np.ndarray, invert: bool = False, transpose: bool = False, rotate: bool = False) -> np.ndarray:
    """Apply transformations to profile required for milling."""

    if invert:
        arr = abs(arr - np.max(arr))

    if transpose:
        arr = arr.T

    if rotate:
        arr = np.rot90(arr)

    return arr

def convert_image_to_bitmap(path: Path, fname: str = "pattern.bmp"):

    # convert image to 24bit uncompressed RGB
    img = Image.open(path).convert('RGB')
    img.save(os.path.join(os.path.dirname(path), fname))

    return img

def convert_profile_to_bmp(arr: np.ndarray) -> np.ndarray:
    # scale values to int
    arr = (arr / np.max(arr)) * 255
    arr = arr.astype(np.uint8)

    # convert to 24bit uncompressed RGB...
    img = Image.fromarray(arr).convert("RGB")

    return np.array(img)

def convert_tif_to_bmp(path: Path, fname: str = "pattern.bmp"):
    arr = np.array(tf.imread(path)).astype(np.float32)

    # scale values to int
    arr = (arr / np.max(arr)) * 255
    arr = arr.astype(np.uint8)

    # convert to 24bit uncompressed RGB...
    img = Image.fromarray(arr).convert("RGB")

    # convert to bmp, save
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(os.path.join(os.path.dirname(path), "helloworld.bmp"))


def save_profile_to_bmp(arr: np.ndarray, path: Path = "profile.bmp"):

    # scale values to int
    arr = (arr / np.max(arr)) * 255
    arr = arr.astype(np.uint8)

    # rescale to 1-255
    arr = arr + (255 - arr) / 255
    arr = arr.astype(np.uint8)

    # convert to 24bit uncompressed RGB...
    img = Image.fromarray(arr).convert("RGB")

    # convert to bmp, save
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return path


### Streamfile utils ###
def calculate_hfw_from_pixel_size(pixel_size: float):
    return (2 ** 16) * pixel_size

def calculate_pixel_size_from_hfw(hfw: float):
    return hfw / (2 ** 16)

def check_protocol(procotol_settings: dict):
    # check existence of pixel_size
    pixel_size = procotol_settings.get("pixel_size")
    if pixel_size is None:
        raise ValueError("Pixel size not set in protocol settings")

    # check existence of max_hfw
    max_hfw = procotol_settings.get("max_hfw")
    if max_hfw is None:
        raise ValueError("Max HFW not set in protocol settings")

    # check pixel_size is less than max_pixel_size from hfw
    max_pixel_size = calculate_pixel_size_from_hfw(procotol_settings.get("max_hfw"))
    if pixel_size > max_pixel_size:
        raise ValueError(f'Pixel size must be less than or equal to {max_pixel_size*1e9}nm, current pixel size is: {pixel_size*1e9}nm')


def calculate_volume_per_second(protocol_settings: dict):
    vpd = protocol_settings.get("vpd")
    if vpd is None:
        raise ValueError("Volume per dose not set in protocol settings")

    milling_current = protocol_settings.get("milling_current")
    if milling_current is None:
        raise ValueError("Milling current not set in protocol settings")

    volume_per_second = vpd * milling_current * 1e9 # from C/s to nC/s
    logging.info(f"Volume per second: {volume_per_second} nC/s")
    return volume_per_second

def calculate_volume_per_dose_base(protocol_settings: dict):
    dose_base = protocol_settings.get("dose_base")
    if dose_base is None:
        raise ValueError("Dose base not set in protocol settings")

    vpt = calculate_volume_per_second(protocol_settings=protocol_settings)
    if vpt is None:
        raise ValueError("Volume per second not calculated correctly")

    volume_per_dose_base = dose_base * vpt
    return volume_per_dose_base


def calculate_percentage_profile(profile):
    return profile / np.sum(profile)


def calculate_n_doses(profile: np.ndarray, protocol_settings: dict):
    depth_sum = np.sum(profile)

    # TODO: make 1e18 explicit (um^3 == m^-18)
    total_volume = depth_sum*(protocol_settings.get('pattern_pixel_size')**2)*1e18
    logging.info(f"Total volume: {np.round(total_volume, 10)} um^3")

    vpdb = calculate_volume_per_dose_base(protocol_settings=protocol_settings)
    logging.info(f"Volume per dose base: {np.round(vpdb, 10)} um^3")

    n_doses = total_volume/vpdb
    logging.info(f"Number of base doses: {int(n_doses):e}")
    return n_doses

def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)