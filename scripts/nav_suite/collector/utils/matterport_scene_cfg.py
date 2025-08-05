# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

MATTERPORT_USD_PATH = "<path-to-your-matterport-usd-file>"
MATTERPORT_PLY_PATH = "<path-to-your-matterport-ply-file>"


def get_matterport_scene_cfg():
    """Get Example Matterport Scene Configuration."""
    # NOTE: wrap in function to avoid import before SimulationApp is initialized

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import patterns
    from isaaclab.utils import configclass

    from nav_suite.sensors import MatterportRayCasterCameraCfg
    from nav_suite.terrains import NavTerrainImporterCfg

    @configclass
    class MatterportSceneCfg(InteractiveSceneCfg):
        """Configuration for the terrain scene with a legged robot."""

        # ground terrain
        terrain = NavTerrainImporterCfg(
            prim_path="/World/Matterport/terrain",
            usd_path=MATTERPORT_USD_PATH,
            terrain_type="usd",
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
            add_colliders=True,
        )

        # rigid object to attach the cameras to
        cube: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )

        # camera snesors
        camera_0 = MatterportRayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/cube",
            mesh_prim_paths=[MATTERPORT_PLY_PATH],
            update_period=0,
            max_distance=10.0,
            data_types=["semantic_segmentation"],
            debug_vis=True,
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=24,
                horizontal_aperture=20.955,
                height=720,
                width=1280,
            ),
        )
        camera_1 = MatterportRayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/cube",
            mesh_prim_paths=[MATTERPORT_PLY_PATH],
            update_period=0,
            max_distance=10.0,
            data_types=["distance_to_image_plane"],
            debug_vis=False,
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=24,
                horizontal_aperture=20.955,
                height=480,
                width=848,
            ),
        )

        # lights
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)),
        )
        sphere_left = AssetBaseCfg(
            prim_path="/World/light_indoor_left",
            spawn=sim_utils.SphereLightCfg(intensity=50000.0, color=(0.75, 0.75, 0.75)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(3.0, 0.0, 3.0)),
        )
        sphere_middle = AssetBaseCfg(
            prim_path="/World/light_indoor_middle",
            spawn=sim_utils.SphereLightCfg(intensity=50000.0, color=(0.75, 0.75, 0.75)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(8.0, 0.0, 3.0)),
        )
        sphere_front = AssetBaseCfg(
            prim_path="/World/light_indoor_front",
            spawn=sim_utils.SphereLightCfg(intensity=50000.0, color=(0.75, 0.75, 0.75)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(8.0, -5.0, 3.0)),
        )
        sphere_front_2 = AssetBaseCfg(
            prim_path="/World/light_indoor_front_2",
            spawn=sim_utils.SphereLightCfg(intensity=50000.0, color=(0.75, 0.75, 0.75)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(8.0, -11.0, 3.0)),
        )
        sphere_right = AssetBaseCfg(
            prim_path="/World/light_indoor_right",
            spawn=sim_utils.SphereLightCfg(intensity=50000.0, color=(0.75, 0.75, 0.75)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(15.0, 0.0, 3.0)),
        )

    return MatterportSceneCfg
