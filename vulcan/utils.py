

import numpy as np
from PIL import Image
import yaml
from pathlib import Path
import os


from enum import Enum


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

def convert_profile_to_bmp(arr: np.ndarray) -> np.ndarray:
    # scale values to int
    arr = (arr / np.max(arr)) * 255
    arr = arr.astype(np.uint8)

    # convert to 24bit uncompressed RGB...
    img = Image.fromarray(arr).convert("RGB")

    return np.array(img)


def save_profile_to_bmp(arr: np.ndarray, path: Path = "profile.bmp"):
    # scale values to int
    arr = (arr / np.max(arr)) * 255
    arr = arr.astype(np.uint8)

    # convert to 24bit uncompressed RGB...
    img = Image.fromarray(arr).convert("RGB")


    # convert to bmp, save
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)