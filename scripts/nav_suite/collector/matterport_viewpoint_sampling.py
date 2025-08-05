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
parser.add_argument("--num_samples", type=int, default=1879, help="Number of samples to sample.")
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
from nav_suite.collectors import ViewpointSampling, ViewpointSamplingCfg
from nav_suite.terrain_analysis import TerrainAnalysisCfg

"""
Main
"""

MatterportSceneCfg = get_matterport_scene_cfg()


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)

    cfg = ViewpointSamplingCfg(
        terrain_analysis=TerrainAnalysisCfg(
            semantic_cost_mapping=os.path.join(NAVSUITE_DATA_DIR, "matterport", "semantic_costs.yaml"),
            raycaster_sensor="camera_0",
        )
    )

    # construct the scene
    scene_cfg = MatterportSceneCfg(args_cli.num_envs, env_spacing=1.0)
    # generate scene
    with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
        scene = InteractiveScene(scene_cfg)
    omni.log.info(f"Scene manager: {scene}")
    with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
        sim.reset()

    explorer = ViewpointSampling(cfg, scene)
    # Now we are ready!
    omni.log.info("Setup complete...")

    # sample and render viewpoints
    samples = explorer.sample_viewpoints(args_cli.num_samples)
    explorer.render_viewpoints(samples)
    print(
        "Viewpoints sampled and rendered will continue to render the environment and visualize the last camera"
        " positions..."
    )

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Simulation loop
    while simulation_app.is_running():
        # Perform step
        sim.render()
        # Update buffers
        explorer.scene.update(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
