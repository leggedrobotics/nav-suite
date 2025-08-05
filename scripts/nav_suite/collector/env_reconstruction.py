# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
This script demonstrates how to use the rigid objects class.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script reconstructs the environment as a 3D pointcloud.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--data_dir", type=str, help="Directory where data of the viewpoint generation is saved.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from nav_suite.utils.environment3d_reconstruction import EnvironmentReconstruction
from nav_suite.utils.environment3d_reconstruction_cfg import ReconstructionCfg

if __name__ == "__main__":
    cfg = ReconstructionCfg()
    cfg.data_dir = args_cli.data_dir

    # start depth reconstruction
    depth_constructor = EnvironmentReconstruction(cfg)
    depth_constructor.depth_reconstruction()

    depth_constructor.save_pcd()
    depth_constructor.show_pcd()
