

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