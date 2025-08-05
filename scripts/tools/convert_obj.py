# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Utility to convert a OBJ into USD format.

This script uses the Asset Converter importer extension from Isaac Sim (``omni.isaac.asset_converter``) to convert a
OBJ asset into USD format. It is designed as a convenience script for command-line use. For more
information on the OBJ importer, see the documentation for the extension:
https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_asset_converter.html


positional arguments:
  input               The path to the input OBJ file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                Show this help message and exit

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a OBJ into USD format.")
parser.add_argument("input", type=str, help="The path to the input OBJ file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict

from nav_suite.utils.obj_converter import ObjConverter
from nav_suite.utils.obj_converter_cfg import ObjConverterCfg


def main():
    # check valid file path
    obj_path = args_cli.input
    if not os.path.isabs(obj_path):
        obj_path = os.path.abspath(obj_path)
    if not check_file_path(obj_path):
        raise ValueError(f"Invalid file path: {obj_path}")
    # create destination path
    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    # Create OBJ converter config
    obj_converter_cfg = ObjConverterCfg(
        asset_path=obj_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        force_usd_conversion=True,
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input OBJ file: {obj_path}")
    print("OBJ importer config:")
    print_dict(obj_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # Create OBJ converter and import the file
    obj_converter = ObjConverter(obj_converter_cfg)
    # print output
    print("OBJ importer output:")
    print(f"Generated USD file: {obj_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)

    # Determine if there is a GUI to update:
    # acquire settings interface
    carb_settings_iface = carb.settings.get_settings()
    # read flag for whether a local GUI is enabled
    local_gui = carb_settings_iface.get("/app/window/enabled")
    # read flag for whether livestreaming GUI is enabled
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # Simulate scene (if not headless)
    if local_gui or livestream_gui:
        # Open the stage with USD
        stage_utils.open_stage(obj_converter.usd_path)
        # Reinitialize the simulation
        app = omni.kit.app.get_app_interface()
        # Run simulation
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                # perform step
                app.update()


if __name__ == "__main__":
    main()
