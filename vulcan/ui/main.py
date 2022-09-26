
import logging
import os
import sys
from pprint import pprint

import napari
import napari.utils.notifications
import vulcan
import vulcan.ui.qtdesigner_files.VulcanUI as VulcanUI
import yaml
from PyQt5 import QtWidgets

BASE_PATH = os.path.dirname(vulcan.__file__)
import os
import traceback
from pathlib import Path
from pprint import pprint

import numpy as np
from autoscript_sdb_microscope_client.structures import (
    BitmapPatternDefinition, StagePosition)
from fibsem import acquire, calibration, constants, milling, movement
from fibsem import utils as fibsem_utils
from fibsem.structures import BeamType, MillingSettings
from vulcan import utils
from vulcan.utils import ChipLocation


class VulcanUI(VulcanUI.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, parent=None, viewer: napari.Viewer = None):
        super().__init__(parent=parent)
        self.setupUi(self)

        self.setWindowTitle("Vulcan UI")
        self.viewer = viewer
        self.base_profile = None

        # microscope, settings
        self.microscope, self.settings = fibsem_utils.setup_session(
            config_path=os.path.join(BASE_PATH, "config"), 
            protocol_path=os.path.join(BASE_PATH, "protocol.yaml"))

        # setup connections
        self.setup_connections()

        # update ui
        self.update_ui_from_config()

        # setup milling
        milling.setup_milling(
            self.microscope, 
            application_file=self.settings.system.application_file,
            hfw=900e-6
        )

        # calibration
        self.calibrated_state = None

    def setup_connections(self):

        # actions
        self.actionLoad_Profile.triggered.connect(self.load_profile)
        self.actionLoad_Configuration.triggered.connect(self.load_configuration)
        self.actionSave_Configuration.triggered.connect(self.save_configuration)

        # buttons
        self.pushButton_move_to_milling_angle.clicked.connect(self.move_to_milling_angle)
        self.pushButton_move_to_chip_location.clicked.connect(self.move_to_chip_location)
        self.pushButton_update_milling_pattern.clicked.connect(self.update_milling_pattern)
        self.pushButton_run_milling.clicked.connect(self.run_milling)
        

        # spinbox
        self.doubleSpinBox_milling_width.valueChanged.connect(self.update_ui)
        self.doubleSpinBox_milling_height.valueChanged.connect(self.update_ui)
        self.doubleSpinBox_milling_depth.valueChanged.connect(self.update_ui)
        self.doubleSpinBox_milling_surface_depth.valueChanged.connect(self.update_ui)

        # checkbox
        self.checkBox_surface_milling.toggled.connect(self.update_ui)
        self.checkBox_profile_invert.toggled.connect(self.update_ui_display)
        self.checkBox_profile_transpose.toggled.connect(self.update_ui_display)
        self.checkBox_profile_rotate.toggled.connect(self.update_ui_display)

        # combobox
        milling_currents = [current for current in self.microscope.beams.ion_beam.beam_current.available_values]
        self.comboBox_milling_current.addItems([f"{current:.3e}" for current in milling_currents])
        self.comboBox_milling_current.currentTextChanged.connect(self.update_ui)

        self.comboBox_chip_location.addItems([location.name for location in ChipLocation])

        # calibration
        self.pushButton_calibration_set_chip_location.clicked.connect(self.set_calibrated_state)
        self.pushButton_calibration_set_edge_location.clicked.connect(self.set_edge_calibration_position)
        self.pushButton_calibration_update_milling_pattern.clicked.connect(self.update_calibration_milling_pattern)
        self.pushButton_calibration_run_milling.clicked.connect(self.run_milling)


    def load_profile(self, profile_filename: Path = None):

        # get file from user
        if profile_filename is None:
            profile_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Load Profile",
                filter="Numpy file (*.npy)",
            )

            if profile_filename == "":
                napari.utils.notifications.show_warning(f"No profile selected.")
                return

        try:
            self.base_profile = utils.load_profile(profile_filename)
        except:
            napari.utils.notifications.show_error(f"Unable to load profile correctly: {traceback.format_exc()}")
            return

        # update ui
        self.profile_filename = profile_filename # save for reloading
        napari.utils.notifications.show_info(f"Profile loaded from: {self.profile_filename}")
        self.label_profile_loaded.setText(f"Profile: {os.path.basename(self.profile_filename)}")
        self.update_ui_display()



    def load_configuration(self):

        config_filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            filter="Yaml config (*.yml *.yaml);;",
        )

        if config_filename == "":
            napari.utils.notifications.show_warning(f"No configuration selected.")
            return

        try:
            self.settings.protocol = fibsem_utils.load_protocol(config_filename)
            self.update_ui_from_config()

        except Exception as e:
            napari.utils.notifications.show_error(f"Unable to load config correctly: {traceback.format_exc()}")

    def save_configuration(self):

        try:
            protocol = self.update_config_from_ui()
        except Exception as e:
            napari.utils.notifications(f"Unable to update config... {traceback.format_exc()}")
            return

        config_filename, ext = QtWidgets.QFileDialog.getSaveFileName(self, 
            caption="Save Configuration", 
            directory=os.path.join(BASE_PATH, "protocol.yaml"), 
            filter="Yaml config (*.yml *.yaml)")

        if config_filename == "":
            napari.utils.notifications.show_warning(f"No filename entered. Configuration was not saved.")
            return

        with open(config_filename, "w") as f:
            yaml.safe_dump(protocol, f, sort_keys=False)

        napari.utils.notifications.show_info(f"Configuration saved successfully ({config_filename}).")

    def update_config_from_ui(self):

        protocol = {}

        # chip
        protocol["chip"] = self.settings.protocol["chip"]

        # profile
        protocol["profile"] = {}
        protocol["profile"]["path"] = self.profile_filename
        protocol["profile"]["invert"] = self.checkBox_profile_invert.isChecked() 
        protocol["profile"]["transpose"] = self.checkBox_profile_transpose.isChecked()
        protocol["profile"]["rotate"] = self.checkBox_profile_rotate.isChecked()

        # milling
        protocol["milling"] = {}
        protocol["milling"]["milling_current"] = float(self.comboBox_milling_current.currentText())
        protocol["milling"]["width"] = float(self.doubleSpinBox_milling_width.value()) * constants.MICRON_TO_METRE
        protocol["milling"]["height"] = float(self.doubleSpinBox_milling_height.value()) * constants.MICRON_TO_METRE
        protocol["milling"]["depth"] = float(self.doubleSpinBox_milling_depth.value()) * constants.MICRON_TO_METRE
        protocol["milling"]["surface_depth"] = float(self.doubleSpinBox_milling_surface_depth.value()) * constants.MICRON_TO_METRE

        return protocol
       
    def update_ui_from_config(self):

        try:
            # system
            self.label_system_plasma_gas_value.setText(str(self.settings.system.ion.plasma_gas))
            self.label_system_application_file_value.setText(str(self.settings.system.application_file))
            self.label_stage_tilt_flat_to_ion_value.setText(str(self.settings.system.stage.tilt_flat_to_ion))
            self.label_stage_rotation_flat_to_ion_value.setText(str(self.settings.system.stage.rotation_flat_to_ion))

            # chip
            self.label_chip_material_value.setText(str(self.settings.protocol["chip"]["material"]))
            self.label_chip_width_value.setText(f'{self.settings.protocol["chip"]["width"]:.2e}')
            self.label_chip_height_value.setText(f'{self.settings.protocol["chip"]["height"]:.2e}')

            # profile
            self.checkBox_profile_invert.setChecked(bool(self.settings.protocol["profile"]["invert"]))
            self.checkBox_profile_transpose.setChecked(bool(self.settings.protocol["profile"]["transpose"]))
            self.checkBox_profile_rotate.setChecked(bool(self.settings.protocol["profile"]["rotate"]))

            profile_path = self.settings.protocol["profile"]["path"]
            if os.path.exists(profile_path):
                self.load_profile(profile_path)

            # milling
            self.doubleSpinBox_milling_width.setValue(float(self.settings.protocol["milling"]["width"]) * constants.METRE_TO_MICRON)
            self.doubleSpinBox_milling_height.setValue(float(self.settings.protocol["milling"]["height"])* constants.METRE_TO_MICRON)
            self.doubleSpinBox_milling_depth.setValue(float(self.settings.protocol["milling"]["depth"]) * constants.METRE_TO_MICRON)
            self.doubleSpinBox_milling_surface_depth.setValue(float(self.settings.protocol["milling"]["surface_depth"]) * constants.METRE_TO_MICRON)

            milling_current = self.settings.protocol['milling']['milling_current']
            self.comboBox_milling_current.setCurrentText(f"{milling_current:.2e}")

        except:
            napari.utils.notifications.show_error(f"Unable to update ui from config: {traceback.format_exc()}")

    def move_to_chip_location(self):

        chip_location = ChipLocation[self.comboBox_chip_location.currentText()]
        logging.info(f"move to chip location: {chip_location}")
        
        if self.calibrated_state is None:
            napari.utils.notifications.show_warning(f"Unable to move to {chip_location.name}. Please set the TopLeft calibrated position in the calibration tab.")
            return

        # chip dimensions
        chip_width = self.settings.protocol["chip"]["width"]
        chip_height = self.settings.protocol["chip"]["height"]


        # move to top left corner
        calibration.set_microscope_state(self.microscope, self.calibrated_state)    
    
        # top left: nothing
        if chip_location is ChipLocation.TopLeft:
            dx, dy = 0, 0
        
        # top right x+= w
        if chip_location is ChipLocation.TopRight:
            dx, dy = chip_width, 0 
        
        # bottom left: y+= h
        if chip_location is ChipLocation.BottomLeft:
            dx, dy = 0, chip_height
        
        # bottom right: x+=w, y+=h
        if chip_location is ChipLocation.BottomRight:
            dx, dy = chip_width, chip_height

        # centre: x+=w/2, y+=h/2
        if chip_location is ChipLocation.Centre:
            dx, dy = chip_width / 2, chip_height / 2

        # relative stage move
        stage_position = StagePosition(x=dx, y=-dy)
        self.microscope.specimen.stage.relative_move(stage_position)

        # TODO: check if this is required / works. think we can just use above
        # movement.move_stage_relative_with_corrected_movement(self.microscope, self.settings, dx=dx, dy=dy, beam_type=BeamType.ION)

        logging.info(f"Location: {chip_location.name}. movement: {stage_position}")

        # TODO: edge detection
        # TODO: offset movements
        # TODO: imaging in napari

    def move_to_milling_angle(self):
        pass 
        # movement.move_flat_to_beam(self.microscope, self.settings, beam_type=BeamType.ION)

    def run_milling(self):


        milling_current = float(self.comboBox_milling_current.currentText())
        milling.run_milling(self.microscope, milling_current, asynch = False)
        logging.info(f"milling_current: {milling_current}")

    def update_ui(self):

        # logging.info(f"update from: {self.sender()}")

        surface_milling_enabled = self.checkBox_surface_milling.isChecked()
        self.label_milling_surface_depth.setVisible(surface_milling_enabled)
        self.doubleSpinBox_milling_surface_depth.setVisible(surface_milling_enabled)
       
    def update_milling_pattern(self):

        # get transformed profile
        profile = self.apply_transfrom_profile()

        # save / load bmp pattern
        bmp_path = os.path.join(BASE_PATH, "tmp", "profile.bmp")
        utils.save_profile_to_bmp(profile, bmp_path)
        bitmap_pattern = BitmapPatternDefinition.load(bmp_path)
        
        # TODO: figure out how to do direct conversion works
        # convert profile to bmp
        # bitmap_pattern = BitmapPatternDefinition()
        # bitmap_pattern.points = utils.convert_profile_to_bmp(profile)


        # milling parameters
        width = self.doubleSpinBox_milling_width.value() * constants.MICRON_TO_METRE
        height = self.doubleSpinBox_milling_height.value() * constants.MICRON_TO_METRE
        depth = self.doubleSpinBox_milling_depth.value() * constants.MICRON_TO_METRE
        surface_depth = self.doubleSpinBox_milling_surface_depth.value() * constants.MICRON_TO_METRE
        surface_milling_enabled = self.checkBox_surface_milling.isChecked()

        self.microscope.imaging.set_active_device(BeamType.ION.value)
        self.microscope.patterning.clear_patterns()
        patterns = []

        # surface milling
        if surface_milling_enabled:
            mill_settings = MillingSettings(width=width, height=height, depth=surface_depth)
            surface_pattern = milling._draw_rectangle_pattern_v2(self.microscope, mill_settings)
            patterns.append(surface_pattern)

        # profile pattern
        mill_settings = MillingSettings(width =width, height=height, depth=depth)
        pattern = milling._draw_bitmap_pattern(self.microscope, mill_settings, bitmap_pattern)
        patterns.append(pattern)

        # TODO: show the milling patterns in napari...
        # need an ion image...
        # display milling time
        # change current before hand

        estimated_time = sum([pattern.time for pattern in patterns])
        # estimated_time = 987654321
        self.label_milling_estimated_time.setText(f"Estimated Time: {estimated_time}")

    def update_ui_display(self):

        if self.base_profile is None:
            return 

        display_profile = self.apply_transfrom_profile()
        
        # update napari viewer
        self.viewer.layers.clear()
        self.viewer.add_image(display_profile, name="profile")

    def apply_transfrom_profile(self) -> np.ndarray:
        # apply users profile transformations

        invert_profile = int(self.checkBox_profile_invert.isChecked())
        transpose_profile = self.checkBox_profile_transpose.isChecked()
        rotate_profile = self.checkBox_profile_rotate.isChecked()

        profile = utils.transform_profile(self.base_profile, 
            invert=invert_profile,
            transpose=transpose_profile,
            rotate=rotate_profile)

        return profile


    ### CALIBRATION
    def set_calibrated_state(self):
        logging.info("set calibrated position")
        self.calibrated_state = calibration.get_current_microscope_state(self.microscope)


    def set_edge_calibration_position(self):
        logging.info(f"set edge calibration position")

        self.edge_calibration_position = calibration.get_current_microscope_state(self.microscope)

    def update_calibration_milling_pattern(self):
        logging.info(f"update calibration milling pattern")

        n_steps = self.spinBox_calibration_steps.value() 
        depth = self.doubleSpinBox_calibration_depth.value() * constants.MICRON_TO_METRE
        spacing = self.doubleSpinBox_calibration_spacing.value() * constants.MICRON_TO_METRE
        width = self.doubleSpinBox_calibration_width.value() * constants.MICRON_TO_METRE
        height = self.doubleSpinBox_calibration_height.value() * constants.MICRON_TO_METRE
        centre_y = -((n_steps - 1) * (height + spacing)) / 2

        logging.info(f"steps: {n_steps}, depth per step: {depth:.2e}, spacing: {spacing:.2e}")     

        self.microscope.patterning.clear_patterns()

        mill_settings = MillingSettings(width=width, height=height, depth=depth, centre_y=centre_y)
        patterns = utils._draw_calibration_patterns(self.microscope, mill_settings, 
                    n_steps=n_steps, offset=spacing)

        estimated_time = sum([pattern.time for pattern in patterns])
        self.label_calibration_estimated_time.setText(f"Estimated Time: {estimated_time}")

        # TODO: show time estimate
        # show run milling button


def main():
    application = QtWidgets.QApplication([])
    viewer = napari.Viewer(ndisplay=2)
    vulcan_ui = VulcanUI(viewer=viewer)                                          
    viewer.window.add_dock_widget(vulcan_ui, area='right')                  
    sys.exit(application.exec_())


if __name__ == "__main__":
    main()
