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

import logging
from fibsem.structures import MillingSettings
from fibsem import milling
def _draw_calibration_patterns(microscope: SdbMicroscopeClient, mill_settings: MillingSettings, n_steps: int = 1 , offset: float = 10e-6) -> list:

    patterns = []
    incremental_depth = mill_settings.depth

    for i in range(n_steps):

        logging.info(f"Step: {i}: settings: {mill_settings}")

        pattern = milling._draw_rectangle_pattern_v2(microscope, mill_settings)
        patterns.append(pattern)

        # update for next pattern
        mill_settings.depth += incremental_depth
        mill_settings.centre_y = mill_settings.centre_y + mill_settings.height / 2 + offset

    return patterns

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

    # TODO: make 1e18 explicit
    total_volume = depth_sum*(protocol_settings.get('pattern_pixel_size')**2)*1e18
    logging.info(f"Total volume: {np.round(total_volume, 10)} um^3")

    vpdb = calculate_volume_per_dose_base(protocol_settings=protocol_settings)
    logging.info(f"Volume per dose base: {np.round(vpdb, 10)} um^3")

    n_doses = total_volume/vpdb
    logging.info(f"Number of base doses: {int(n_doses):e}")
    return n_doses