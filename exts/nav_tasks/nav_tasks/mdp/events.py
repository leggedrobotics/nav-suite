# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import omni.log
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

from nav_suite.terrain_analysis import TerrainAnalysisCfg, TerrainAnalysisSingletonCfg

from .commands import FixGoalCommand, GoalCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


###
# Terrain Analysis based Reset
###


class TerrainAnalysisRootReset:

    def __init__(self, cfg: TerrainAnalysisCfg | TerrainAnalysisSingletonCfg, robot_dim: float = 0.5):
        self.cfg = cfg

        # get the robot dimension w.r.t. to the height map resolution
        self.robot_idx_dim = math.ceil(robot_dim / self.cfg.grid_resolution)

    def _run_analysis(self, env: ManagerBasedRLEnv):
        """Run the terrain analysis to compute the root state reset."""
        # check if singleton is used and available
        if hasattr(self.cfg.class_type, "instance") and self.cfg.class_type.instance() is not None:
            self.analyser = self.cfg.class_type.instance()
        else:
            self.analyser = self.cfg.class_type(self.cfg, env.scene)
        omni.log.info("Running terrain analysis")
        self.analyser.analyse()
        omni.log.info("Terrain analysis completed")

    def _get_spawn_height(self, positions: torch.Tensor) -> torch.Tensor:
        """Get the spawn height for the robot based on the terrain analysis."""
        # get the index of each position in the height map
        pos_idx = (
            (
                positions[..., :2]
                - torch.tensor(
                    [self.analyser.mesh_dimensions[2], self.analyser.mesh_dimensions[3]], device=positions.device
                )
            )
            / self.cfg.grid_resolution
        ).int()
        # limit the position index to boundaries of the map
        pos_idx[:, 0] = torch.clamp(
            pos_idx[:, 0], self.robot_idx_dim, self.analyser.height_grid.shape[0] - 1 - self.robot_idx_dim
        )
        pos_idx[:, 1] = torch.clamp(
            pos_idx[:, 1], self.robot_idx_dim, self.analyser.height_grid.shape[1] - 1 - self.robot_idx_dim
        )
        # get the part of the height map for each robot
        local_height_map_max = [
            torch.max(
                self.analyser.height_grid[
                    curr_pos_idx[0] - self.robot_idx_dim : curr_pos_idx[0] + self.robot_idx_dim,
                    curr_pos_idx[1] - self.robot_idx_dim : curr_pos_idx[1] + self.robot_idx_dim,
                ]
            )
            for curr_pos_idx in pos_idx
        ]
        # return the maximum height of the local height map for each robot
        return torch.stack(local_height_map_max).to(positions.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        yaw_range: tuple[float, float],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        """The function samples the new start positions in the terrain with random initial velocities."""

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        # get default root state
        root_states = asset.data.default_root_state[env_ids].clone()

        # positions
        if not hasattr(self, "analyser"):
            self._run_analysis(env)
        positions = self.analyser.points[torch.randperm(self.analyser.points.shape[0])[: len(env_ids)]]

        # the spawn point does not consider the surrounding terrain, which can lead to robot being spawned inside
        # the terrain. To avoid this, we do an additional step here with a safety margin of the grid resolution
        # with a maximum of 0.3m
        positions[:, 2] = (
            self._get_spawn_height(positions)
            + asset.data.default_root_state[env_ids, 2]
            + max(self.analyser.cfg.grid_resolution * 2, 0.3)
        )

        # yaw range
        yaw_samples = sample_uniform(yaw_range[0], yaw_range[1], (len(env_ids), 1), device=asset.device)
        orientations = quat_from_euler_xyz(
            torch.zeros_like(yaw_samples), torch.zeros_like(yaw_samples), yaw_samples
        ).squeeze(1)

        # velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

        velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

        # obtain default joint positions
        default_joint_pos = asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

    def __name__(self):
        # return the name of the function for logging purposes
        return "TerrainAnalysisRootReset"


def reset_robot_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    yaw_range: tuple[float, float],
    velocity_range: dict[str, tuple[float, float]] = {},
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_command_generator_name: str = "goal_command",
    spawn_in_env_frame: bool = False,
    set_default_joint_state: bool = True,
    add_default_base_pos: bool = False,
):
    """Reset the asset root state to the spawn state defined by the command generator.

    Args:
        env: The environment object.
        env_ids: The environment ids to reset.
        yaw_range: The additive heading range to apply to the spawn heading.
        velocity_range: The velocity range to apply to the spawn velocity.
        asset_cfg: The asset configuration to reset. Defaults to SceneEntityCfg("robot").
        spawn_in_env_frame: Whether spawn is based on the environment frame. Defaults to False.
        set_default_joint_state: Whether to set the default joint state. Defaults to True.
        add_default_base_pos: Whether to add the default base position to the spawn position. Defaults to False.
            This is typically not necessary, as the robot height is added to the points in the
            :class:`nav_collectors.terrain_analysis.TerrainAnalysis` Module

    .. note::
        This event assumes the existence of 'pos_spawn_w' and 'heading_spawn_w' in the command generator term,
        i.e. an goal command generator of type :class:`nav_tasks.mdp.GoalCommand` is required.
        See :class:`nav_tasks.mdp.TerrainAnalysisRootReset` for a more general approach.

    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_generator: GoalCommand | FixGoalCommand = env.command_manager.get_term(goal_command_generator_name)  # type: ignore

    # positions - based on given start points (command generator)
    positions = goal_cmd_generator.pos_spawn_w[env_ids]
    if add_default_base_pos:
        positions += asset.data.default_root_state[env_ids, :3]
    if spawn_in_env_frame:
        positions += env.scene.env_origins[env_ids]

    # yaw range
    yaw_samples = sample_uniform(yaw_range[0], yaw_range[1], (len(env_ids), 1), device=asset.device)
    yaw_samples += goal_cmd_generator.heading_spawn_w[env_ids].unsqueeze(1)
    orientations = quat_from_euler_xyz(
        torch.zeros_like(yaw_samples), torch.zeros_like(yaw_samples), yaw_samples
    ).squeeze(1)

    # velocities - zero
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = asset.data.default_root_state[env_ids, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    if set_default_joint_state:
        # obtain default joint positions
        default_joint_pos = asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
