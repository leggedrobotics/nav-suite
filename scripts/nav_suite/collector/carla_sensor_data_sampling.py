

"""
This script demonstrates how to use the rigid objects class.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
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
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.timer import Timer

from nav_suite import NAVSUITE_DATA_DIR
from nav_suite.collectors import CameraSensorCfg, SensorDataSampling, SensorDataSamplingCfg
from nav_suite.terrains import NavTerrainImporterCfg

"""
Main
"""


@configclass
class TestTerrainCfg(InteractiveSceneCfg):
    """Configuration for a matterport terrain scene with a camera."""

    # ground terrain
    terrain = NavTerrainImporterCfg(
        prim_path="/World/Carla",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        usd_path=os.path.join(NAVSUITE_DATA_DIR, "unreal", "town01", "carla_export", "carla.usd"),
        terrain_type="usd",
        duplicate_cfg_file=[
            os.path.join(NAVSUITE_DATA_DIR, "unreal", "town01", "cw_multiply_cfg.yml"),
            os.path.join(NAVSUITE_DATA_DIR, "unreal", "town01", "vehicle_cfg.yml"),
        ],
        sem_mesh_to_class_map=os.path.join(NAVSUITE_DATA_DIR, "unreal", "town01", "keyword_mapping.yml"),
        people_config_file=os.path.join(NAVSUITE_DATA_DIR, "unreal", "town01", "people_cfg.yml"),
    )
    # camera
    camera_0 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/sem_cam",
        update_period=0,
        data_types=["semantic_segmentation"],
        debug_vis=True,
        offset=CameraCfg.OffsetCfg(pos=(0.419, -0.025, -0.020), rot=(0.992, 0.008, 0.127, 0.001), convention="world"),
        height=720,
        width=1280,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24,
            horizontal_aperture=20.955,
        ),
    )
    camera_1 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/depth_cam",
        update_period=0,
        data_types=["distance_to_image_plane"],
        debug_vis=False,
        offset=CameraCfg.OffsetCfg(pos=(0.419, -0.025, -0.020), rot=(0.992, 0.008, 0.127, 0.001), convention="world"),
        height=480,
        width=848,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24,
            horizontal_aperture=20.955,
        ),
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
    sim.set_camera_view([130, -125, 30], [100, -130, 0.5])

    # setup sampling config
    cfg = SensorDataSamplingCfg(sensor_data_handlers=[CameraSensorCfg()])
    # overwrite semantic cost mapping and adjust parameters based on larger map
    cfg.terrain_analysis.semantic_cost_mapping = os.path.join(
        NAVSUITE_DATA_DIR, "unreal", "town01", "semantic_costs.yaml"
    )
    cfg.terrain_analysis.grid_resolution = 1.0
    cfg.terrain_analysis.sample_points = 10000
    # limit space to be within the road network
    cfg.terrain_analysis.dim_limiter_prim = "Road_Sidewalk"
    # enable debug visualization
    cfg.terrain_analysis.viz_graph = True

    # override the scene configuration
    scene_cfg = TestTerrainCfg(args_cli.num_envs, env_spacing=1.0)
    # generate scene
    with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
        scene = InteractiveScene(scene_cfg)
    omni.log.info(f"Scene manager: {scene}")
    with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
        sim.reset()

    data_sampler = SensorDataSampling(cfg, scene)
    # Now we are ready!
    omni.log.info("Setup complete...")

    # sample and render sensor data
    samples = data_sampler.sample_sensor_data(9560)
    data_sampler.render_sensor_data(samples)
    print(
        "Sensor data sampled and rendered will continue to render the environment and visualize the last camera"
        " positions..."
    )

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Simulation loop
    while simulation_app.is_running():
        # Perform step
        sim.render()
        # Update buffers
        data_sampler.scene.update(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
