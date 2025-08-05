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
from utils import get_matterport_scene_cfg

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os

import isaaclab.sim as sim_utils
import omni.log
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.timer import Timer

from nav_suite import NAVSUITE_DATA_DIR
from nav_suite.collectors import TrajectorySampling, TrajectorySamplingCfg
from nav_suite.terrain_analysis import TerrainAnalysisCfg

MatterportSceneCfg = get_matterport_scene_cfg()

"""
Main
"""


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([10.0, 1.5, 2.0], [8.0, -1.0, 0.5])

    cfg = TrajectorySamplingCfg(
        terrain_analysis=TerrainAnalysisCfg(
            semantic_cost_mapping=os.path.join(NAVSUITE_DATA_DIR, "matterport", "semantic_costs.yaml"),
            raycaster_sensor="camera_0",
        )
    )
    cfg.terrain_analysis.viz_graph = True

    # construct the scene
    scene_cfg = MatterportSceneCfg(args_cli.num_envs, env_spacing=1.0)
    # generate scene
    with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
        scene = InteractiveScene(scene_cfg)
    omni.log.info(f"Scene manager: {scene}")
    with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
        sim.reset()

    explorer = TrajectorySampling(cfg, scene)
    # Now we are ready!
    omni.log.info("Setup complete...")

    # sample viewpoints
    explorer.sample_paths(1000, 0.5, 10.0)

    print("Trajectories sampled and simulation will continue to render the environment...")
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # get default cube positions
    default_cube_pose = explorer.scene.rigid_objects["cube"].data.default_root_state
    # Simulation loop
    while simulation_app.is_running():
        # set cube position
        explorer.scene.rigid_objects["cube"].write_root_state_to_sim(default_cube_pose)
        explorer.scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Update buffers
        explorer.scene.update(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
