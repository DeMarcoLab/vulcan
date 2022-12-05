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
    profile += 0.1e-6

    x, y = create_position_grid(profile=profile, protocol_settings=protocol_settings)
    logging.info(f"Position grid created successfully")

    percentage_profile = utils.calculate_percentage_profile(profile)

    # calculate parameters
    n_doses = utils.calculate_n_doses(profile=profile, protocol_settings=protocol_settings)

    dose_profile = percentage_profile * n_doses
    dt_profile = dose_profile/protocol_settings.get("n_passes")

    scale_factor = 1.2136
    stream_list = []
    for i in range(dt_profile.shape[0]):
        for j in range(dt_profile.shape[1]):
            if i == 0 and j == 0:
                print(dt_profile[i, j]/scale_factor, 1)
            stream_list.append([int(np.round(dt_profile[i, j]/scale_factor)), int(np.round(x[i, j])), int(np.round(y[i, j]))])
    logging.info(f"Stream list created successfully")

    if protocol_settings.get("shuffle"):
        np.random.shuffle(stream_list)
        logging.info(f"Stream list shuffled")

    n_passes = protocol_settings.get("n_passes")
    save_path = protocol_settings.get("save_path")

    with open(f"{save_path}.str", "w") as f:
        # f.write("s16,25ns\n")
        f.write("s16\n")
        f.write(str(int(n_passes))+"\n")
        f.write(str(profile.shape[0]*profile.shape[1])+"\n")
        for position in stream_list:
            f.write(f"{position[0]} {position[1]} {position[2]}\n")

        logging.info(f"Stream file saved to {save_path}.str")

    return f"{save_path}.str"


def send_to_microscope(filename: str,
                       microscope: SdbMicroscopeClient,
                       protocol_settings: dict):

    microscope = SdbMicroscopeClient()
    microscope.connect("10.0.0.1")

    microscope.patterning.set_default_beam_type(BeamType.ION)
    microscope.patterning.set_default_application_file("Si")

    microscope.beams.ion_beam.horizontal_field_width.value = utils.calculate_hfw_from_pixel_size(protocol_settings.get("pixel_size"))

    if protocol_settings.get("clear"):
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
    convert_arr_to_streamfile(profile, settings.protocol)

    if settings.protocol.get("send"):
        save_name  = f"{settings.protocol.get('save_path')}.str"
        send_to_microscope(filename=save_name, microscope=microscope, protocol_settings=settings.protocol)


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

if __name__ == "__main__":
    # # profile = np.ones(shape=(201, 201)) * 5.e-6
    # # np.save("recreation_94.npy", profile)

    # profile = np.load(r'C:\Users\Admin\Github\vulcan\vulcan\profiles\sub_um_str.npy')
    # print(profile.shape)
    # import matplotlib.pyplot as plt
    # profile = profile[:400, :]
    # # pad profile by 10 % on either side
    # print(profile.shape)
    # profile = np.pad(profile, int(profile.shape[0]*0.05), mode="constant", constant_values=0)
    # # save as sub_um_str_pad.npy
    # np.save(r'C:\Users\Admin\Github\vulcan\vulcan\profiles\sub_um_str_pad.npy', profile)
    # plt.imshow(profile)
    # plt.show()

    if len(sys.argv) < 3:
        raise ValueError("Must provide experiement name and profile path")
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    # TODO: figure out if better way to do this