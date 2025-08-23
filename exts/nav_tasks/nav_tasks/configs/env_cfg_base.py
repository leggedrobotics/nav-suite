# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

from nav_suite.collectors import TrajectorySamplingCfg
from nav_suite.terrain_analysis import TerrainAnalysisCfg

import nav_tasks.mdp as mdp
from nav_tasks.sensors import ZED_X_MINI_WIDE_RAYCASTER_CFG, adjust_ray_caster_camera_image_size

# Reset cuda memory
torch.cuda.empty_cache()

TERRAIN_MESH_PATH: list[str | RayCasterCfg.RaycastTargetCfg] = ["/World/ground"]

IMAGE_SIZE_DOWNSAMPLE_FACTOR = 15


##
# Scene definition
##


@configclass
class NavTasksDepthNavSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene for training a perceptive navigation policy on an AnymalD Robot."""

    # TERRAIN
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=True,
    )

    # ROBOTS
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # SENSORS: Locomotion Policy
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # SENSORS: Navigation Policy
    front_zed_camera = ZED_X_MINI_WIDE_RAYCASTER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=TERRAIN_MESH_PATH,
        update_period=0,
        debug_vis=False,
        offset=RayCasterCameraCfg.OffsetCfg(
            # The camera can be mounted at either 10 or 15 degrees on the robot.
            # pos=(0.4761, 0.0035, 0.1055), rot=(0.9961947, 0.0, 0.087155, 0.0), convention="world"  # 10 degrees
            pos=(0.4761, 0.0035, 0.1055),
            rot=(0.9914449, 0.0, 0.1305262, 0.0),
            convention="world",  # 15 degrees
        ),
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # LIGHTS
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)),
    )

    def __post_init__(self):
        """Post initialization."""
        # Downsample the camera data to a usable size for the project.
        self.front_zed_camera = adjust_ray_caster_camera_image_size(
            self.front_zed_camera, IMAGE_SIZE_DOWNSAMPLE_FACTOR, IMAGE_SIZE_DOWNSAMPLE_FACTOR
        )

        # turn off the self-collisions
        self.robot.spawn.articulation_props.enabled_self_collisions = False


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    velocity_command = mdp.NavigationSE2ActionCfg(
        asset_name="robot",
        low_level_action=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        ),
        low_level_policy_file=ISAACLAB_NUCLEUS_DIR + "/Policies/ANYmal-C/HeightScan/policy.pt",
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class LocomotionPolicyCfg(ObsGroup):
        """
        Observations for locomotion policy group. These are fixed when training a navigation
        policy using a pre-trained locomotion policy.
        """

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.vel_commands, params={"action_term": "velocity_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_low_level_action, params={"action_term": "velocity_command"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class NavigationPolicyCfg(ObsGroup):
        """Observations for navigation policy group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        goal_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_command"})

        forwards_depth_image = mdp.EmbeddedDepthImageCfg(
            sensor_cfg=SceneEntityCfg("front_zed_camera"),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Observation Groups
    low_level_policy: LocomotionPolicyCfg = LocomotionPolicyCfg()
    policy: NavigationPolicyCfg = NavigationPolicyCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    reset_base = EventTerm(
        func=mdp.reset_robot_position,
        mode="reset",
        params={
            "yaw_range": (-3.0, 3.0),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    .. note::
        All rewards get multiplied with weight*dt - consider this when setting weights.
        Rewards are normalized over max episode length in wandb logging.

    .. note::
        In wandb:
        - Episode Rewards are in seconds
        - Train Mean Reward is based on episode length (Rewards * Episode Length)
    """

    # -- rewards
    # Sparse: only when the "stayed_at_goal" condition is met, per the goal_reached term in TerminationsCfg
    goal_reached_rew = RewTerm(
        func=mdp.is_terminated_term,  # returns 1 if the goal is reached and env has NOT timed out # type: ignore
        params={"term_keys": "goal_reached"},
        weight=1000.0,  # make it big
    )

    stepped_goal_progress = mdp.SteppedProgressCfg(
        step=0.05,
        weight=1.0,
    )
    near_goal_stability = RewTerm(
        func=mdp.near_goal_stability,
        weight=2.0,  # Dense Reward of [0.0, 0.1] --> Max Episode Reward: 1.0
    )
    near_goal_angle = RewTerm(
        func=mdp.near_goal_angle,
        weight=1.0,  # Dense Reward of [0.0, 0.025]  --> Max Episode Reward: 0.25
    )

    # -- penalties
    lateral_movement = RewTerm(
        func=mdp.lateral_movement,
        weight=-0.1,  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )
    backward_movement = RewTerm(
        func=mdp.backwards_movement,
        weight=-0.1,  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )
    episode_termination = RewTerm(
        func=mdp.is_terminated_term,  # type: ignore
        params={"term_keys": ["base_contact", "leg_contact"]},
        weight=-200.0,  # Sparse Reward of {-20.0, 0.0} --> Max Episode Penalty: -20.0
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.1,  # Dense Reward of [-0.01, 0.0] --> Max Episode Penalty: -0.1
    )


@configclass
class TerminationsCfg:
    """
    Termination terms for the MDP.

    .. note::
        The time_out flag behavior:
        - If set to True, no termination penalty is added
        - In the RSL_RL library, time_out=True affects reward handling before optimization
        - When time_out=True, rewards are bootstrapped

    .. note::
        Wandb Episode Termination metrics are independent of:
        - Number of robots
        - Episode length
        - Other environment parameters
    """

    time_out = DoneTerm(
        func=mdp.proportional_time_out,
        params={
            "max_speed": 1.0,
            "safety_factor": 4.0,
        },
        time_out=True,  # No termination penalty for time_out = True
    )

    goal_reached = DoneTerm(
        func=mdp.StayedAtGoal,  # type: ignore
        params={
            "time_threshold": 2.0,
            "distance_threshold": 0.5,
            "angle_threshold": 0.3,
            "speed_threshold": 0.6,
        },
        time_out=False,
    )

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]),
            "threshold": 0.0,
        },
        time_out=False,
    )

    leg_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*THIGH", ".*HIP", ".*SHANK"]),
            "threshold": 0.0,
        },
        time_out=False,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.
    NOTE: steps = learning_iterations * num_steps_per_env)
    """

    initial_heading_pertubation = CurrTerm(
        func=mdp.modify_heading_randomization_linearly,
        params={
            "event_term_name": "reset_base",
            "perturbation_range": (0.0, 3.0),
            "step_range": (0, 500 * 48),
        },
    )

    goal_conditions_ramp = CurrTerm(
        func=mdp.modify_goal_conditions,
        params={
            "termination_term_name": "goal_reached",
            "time_range": (2.0, 2.0),
            "distance_range": (1, 0.5),
            "angle_range": (0.6, 0.3),
            "speed_range": (1.3, 0.6),
            "step_range": (0, 500 * 48),
        },
    )

    # Increase goal distance & resample trajectories
    goal_distances = CurrTerm(
        func=mdp.modify_goal_distance_in_steps,
        params={
            "update_rate_steps": 100 * 48,
            "min_path_length_range": (0.0, 2.0),
            "max_path_length_range": (5.0, 15.0),
            "step_range": (50 * 48, 1500 * 48),
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    goal_command = mdp.GoalCommandCfg(
        asset_name="robot",
        z_offset_spawn=0.1,
        num_pairs=1000,
        path_length_range=[2.0, 10.0],
        traj_sampling=TrajectorySamplingCfg(
            enable_saved_paths_loading=False,
            terrain_analysis=TerrainAnalysisCfg(
                raycaster_sensor="front_zed_camera",
                max_terrain_size=100.0,
                semantic_cost_mapping=None,
                viz_graph=False,
                viz_height_map=False,
            ),
        ),
        resampling_time_range=(1.0e9, 1.0e9),  # No resampling
        debug_vis=True,
    )


##
# Environment configuration
##


@configclass
class NavTasksDepthNavEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # Scene settings
    scene: NavTasksDepthNavSceneCfg = NavTasksDepthNavSceneCfg(num_envs=100, env_spacing=8)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    viewer: ViewerCfg = ViewerCfg()

    def __post_init__(self):
        """Post initialization."""

        # Simulation settings
        self.sim.dt = 0.005  # In seconds
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # General settings
        self.episode_length_s = 20

        # This sets how many times the high-level actions (navigation policy)
        # are applied to the sim before being recalculated.
        # self.sim.dt * self.decimation = 0.005 * 20 = 0.1 seconds -> 10Hz.
        self.fz_low_level_planner = 10  # Hz
        self.decimation = int(200 / self.fz_low_level_planner)

        # Similar to above, the low-level actions (locomotion controller) are calculated every:
        # self.sim.dt * self.low_level_decimation, so 0.005 * 4 = 0.02 seconds, or 50Hz.
        self.low_level_decimation = 4

        # update sensor update periods
        # We tick contact sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # We tick the cameras based on the navigation policy update period.
        if self.scene.front_zed_camera is not None:
            self.scene.front_zed_camera.update_period = self.decimation * self.sim.dt


######################################################################
# Anymal D - TRAIN & PLAY & DEV Configuration Modifications
######################################################################


@configclass
class NavTasksDepthNavEnvCfg_TRAIN(NavTasksDepthNavEnvCfg):
    def __post_init__(self):
        super().__post_init__()


@configclass
class NavTasksDepthNavEnvCfg_PLAY(NavTasksDepthNavEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Change number of environments
        self.scene.num_envs = 10

        # Disable curriculum
        self.curriculum = CurriculumCfg()

        # Set fixed parameters for play mode
        self.events.reset_base.params["yaw_range"] = (0, 0)
        self.terminations.goal_reached.params = {
            "time_threshold": 0.1,
            "distance_threshold": 0.5,
            "angle_threshold": 0.3,
            "speed_threshold": 0.6,
        }

        self.viewer.eye = (0.0, 7.0, 7.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class NavTasksDepthNavEnvCfg_DEV(NavTasksDepthNavEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 2
