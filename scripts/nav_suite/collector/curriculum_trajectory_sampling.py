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

import itertools
import numpy as np

import isaaclab.sim as sim_utils
import omni.log
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.timer import Timer

from nav_suite.collectors import TrajectorySampling, TrajectorySamplingCfg
from nav_suite.terrain_analysis import TerrainAnalysisCfg

"""
Main
"""


@configclass
class TestTerrainCfg(InteractiveSceneCfg):
    """Configuration for a matterport terrain scene with a camera."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/Terrain",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(num_rows=4, num_cols=6, border_width=1),
    )
    # camera
    scanner = RayCasterCfg(
        prim_path="/World/Terrain",
        update_period=0,
        debug_vis=False,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(1.0, 1.0),
        ),
        mesh_prim_paths=["/World/Terrain"],
        ray_alignment="yaw",
    )
    # extras - light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
    )


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([-25, 0, 30], [0, 0, 0.0])

    # setup config
    cfg = TrajectorySamplingCfg(
        terrain_analysis=TerrainAnalysisCfg(raycaster_sensor="scanner", viz_height_map=False, viz_graph=True),
    )
    # enable debug visualization
    cfg.terrain_analysis.viz_graph = True

    # get scene cfg
    scene_cfg = TestTerrainCfg(args_cli.num_envs, env_spacing=1.0)
    # generate scene
    with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
        scene = InteractiveScene(scene_cfg)
    omni.log.info(f"Scene manager: {scene}")
    with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
        sim.reset()

    explorer = TrajectorySampling(cfg, scene)
    # Now we are ready!
    omni.log.info("Setup complete...")

    # sample trajectories
    traj = explorer.sample_paths_by_terrain(1000, 2.0, 10.0)

    omni.log.info(": Trajectories sampled and simulation will continue to render the environment...")

    # visualize the trajectories using debug draw
    try:
        from isaacsim.util.debug_draw import _debug_draw as omni_debug_draw

        draw_interface = omni_debug_draw.acquire_debug_draw_interface()

        colors = np.random.rand(traj.shape[0], traj.shape[1], 4)
        colors[..., 3] = 1.0

        for row, col in itertools.product(range(traj.shape[0]), range(traj.shape[1])):
            draw_interface.draw_lines(
                traj[row, col, :, 0:3].tolist(),
                traj[row, col, :, 3:6].tolist(),
                [tuple(colors[row, col].tolist())] * traj.shape[2],
                [5] * traj.shape[2],
            )

    except ImportError:
        omni.log.info("Debug draw is not available. To viz the trajectory, run in non-headless.")

    # Simulation loop
    while simulation_app.is_running():
        # Perform step
        sim.render()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
