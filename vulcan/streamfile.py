import logging
import os
import sys

import fibsem.utils as f_utils
import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import *
from autoscript_sdb_microscope_client.structures import *

from vulcan import utils


def convert_arr_to_streamfile(profile: np.ndarray,
                              protocol_settings: dict):

    # read in protocol settings
    pixel_size = protocol_settings.get("pattern_pixel_size")

    pattern_width = profile.shape[0] * pixel_size
    pattern_height = profile.shape[1] * pixel_size

    # basic checks on pattern
    if not pattern_width <= 800:
        logging.error("Pattern width must be less than or equal to 800 um")
        raise ValueError("Pattern width must be less than or equal to 800 um")
    if not pattern_height <= 500:
        logging.error("Pattern height must be less than or equal to 500 um")
        raise ValueError("Pattern height must be less than or equal to 500 um")

    if protocol_settings.get("invert"):
        profile = profile.max() - profile
        logging.info(f"Profile inverted")

    x, y = create_position_grid(profile=profile, protocol_settings=protocol_settings)
    logging.info(f"Position grid created successfully")

    percentage_profile = utils.calculate_percentage_profile(profile)

    # calculate parameters
    n_doses = utils.calculate_n_doses(profile=profile, protocol_settings=protocol_settings)

    dose_profile = percentage_profile * n_doses
    dt_profile = dose_profile/protocol_settings.get("n_passes")

    stream_list = []
    for i in range(dt_profile.shape[0]):
        for j in range(dt_profile.shape[1]):
            stream_list.append([dt_profile[i, j], x[i, j], y[i, j]])
    logging.info(f"Stream list created successfully")

    if protocol_settings.get("shuffle"):
        np.random.shuffle(stream_list)
        logging.info(f"Stream list shuffled")

    n_passes = protocol_settings.get("n_passes")
    save_path = protocol_settings.get("save_path")

    with open(f"{save_path}.str", "w") as f:
        f.write("s16\n")
        f.write(str(int(n_passes))+"\n")
        f.write(str(profile.shape[0]*profile.shape[1])+"\n")
        for position in stream_list:
            f.write(f"{position[0]} {position[1]} {position[2]}\n")

        logging.info(f"Stream file saved to {save_path}.str")

    return f"{save_path}.str"


def send_to_microscope(filename: str,
                       microscope: SdbMicroscopeClient,
                       ):

    microscope = SdbMicroscopeClient()
    microscope.connect("10.0.0.1")

    microscope.patterning.set_default_beam_type(BeamType.ION)
    microscope.patterning.set_default_application_file("Si")

    microscope.patterning.clear_patterns()

    spd = StreamPatternDefinition.load(filename)
    microscope.patterning.create_stream(0, 0, spd)
# TODO: Add merging/multi-step?


def main(experiment_name: str,
         profile_path: str,
         config_dir: str = "config",
         config_path: str = "config/protocol.yaml",
         ):

    microscope, settings = f_utils.setup_session(experiment_name, config_dir, config_path)

    utils.check_protocol(procotol_settings=settings.protocol)
    logging.info(f"Valid protocol settings")
    logging.info(f"HFW: {utils.calculate_hfw_from_pixel_size(settings.protocol.get('pixel_size'))}")

    profile = np.load(profile_path)
    streamfile = convert_arr_to_streamfile(profile, settings.protocol)

    if settings.protocol.get("send"):
        send_to_microscope(filename=save_filename, microscope=microscope)


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

    # divide the meshgrid by the pixel_size
    x = np.round(x / pixel_size)
    y = np.round(y / pixel_size)

    return x, y

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Must provide experiement name and profile path")
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    # TODO: figure out if better way to do this