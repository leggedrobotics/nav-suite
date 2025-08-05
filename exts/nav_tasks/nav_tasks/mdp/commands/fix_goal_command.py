# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-module containing command generators for the position-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse, sample_uniform, yaw_quat

from nav_suite.terrain_analysis import TerrainAnalysis, TerrainAnalysisSingleton

from .goal_command_base import GoalCommandBaseTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .fix_goal_command_cfg import FixGoalCommandCfg


class FixGoalCommand(GoalCommandBaseTerm):
    r"""Fixed goal command generator.

    The goals are provided relative to the terrain origin. The goal is fixed and does not change during the episode and
    is the same for all environments. The goal is provided in the world frame and is transformed to the base frame of
    the robot.
    """

    cfg: FixGoalCommandCfg
    """Configuration for the command."""

    def __init__(self, cfg: FixGoalCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command class.

        Args:
            cfg: The configuration parameters for the command.
            env: The environment object.
        """
        super().__init__(cfg, env)
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # get the relative location of the terrain origin from which the goal position is set
        env_terrain_center = torch.zeros((self.num_envs, 3), device=self.device)
        if self.cfg.relative_terrain_origin == "center" and env.scene.terrain.cfg.terrain_type == "generator":
            # check if terrain origins are defined
            assert env.scene.terrain.terrain_origins is not None, "Terrain origins are not defined."
            # compute the terrain center of each subterrain for each environment
            # necessary as the env_origin does not have to align with the terrain center
            subterrain_shape = env.scene.terrain.terrain_origins.shape[:2]
            subterrain_size = env.scene.terrain.cfg.terrain_generator.size
            grid_x, grid_y = torch.meshgrid(
                torch.arange(
                    (1 - subterrain_shape[0]) * subterrain_size[0] / 2,
                    subterrain_size[0] * (subterrain_shape[0]) / 2,
                    subterrain_size[0],
                    device=self.device,
                ),
                torch.arange(
                    (1 - subterrain_shape[1]) * subterrain_size[1] / 2,
                    subterrain_size[1] * (subterrain_shape[1]) / 2,
                    subterrain_size[1],
                    device=self.device,
                ),
            )
            terrain_center = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)

            # map env_origins to closest terrain center
            env_origins = env.scene.env_origins[:, :2]
            env_origins = env_origins.unsqueeze(1).expand(-1, subterrain_shape[0] * subterrain_shape[1], -1)
            terrain_center_expand = terrain_center.unsqueeze(0).expand(env_origins.shape[0], -1, -1)
            env_terrain_center[:, :2] = terrain_center[
                torch.argmin(torch.norm(env_origins - terrain_center_expand, dim=-1), dim=-1)
            ]
            env_terrain_center[:, 2] = env.scene.env_origins[:, 2]

        else:
            # use the defined terrain origin which can be shifted from the actual terrain center
            env_terrain_center[:] = env.scene.env_origins

        # -- goal commands: (x, y, z)
        self.pos_command_w = env_terrain_center + torch.tensor(self.cfg.fix_goal_position, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        # perturb the goal positions
        self.pos_command_w += torch.hstack([
            sample_uniform(self.cfg.goal_rand_x[0], self.cfg.goal_rand_x[1], (self.num_envs, 1), device=self.device),
            sample_uniform(self.cfg.goal_rand_y[0], self.cfg.goal_rand_y[1], (self.num_envs, 1), device=self.device),
            torch.zeros((self.num_envs, 1), device=self.device),
        ])

        # -- spawn locations (x, y, z, heading)
        self.pos_spawn_w = env.scene.env_origins.clone()
        self.heading_spawn_w = torch.zeros(self.num_envs, device=self.device)

        self.path_length_command = torch.zeros(self.num_envs, device=self.device)

        # get the height map of the terrain to elevate the goal positions
        if self.cfg.project_onto_terrain:
            # check if singleton is used and available
            if (
                hasattr(self.cfg.terrain_analysis.class_type, "instance")
                and self.cfg.terrain_analysis.class_type.instance() is not None
            ):
                self._analysis = self.cfg.terrain_analysis.class_type.instance()
            else:
                self._analysis = self.cfg.terrain_analysis.class_type(self.cfg.terrain_analysis, scene=self._env.scene)
            self.pos_command_w[:, 2] += self._analysis.get_height(self.pos_command_w[:, :2])

        # EVAL case that maximum number of samples is set
        if self.cfg.trajectory_num_samples is not None:
            self.all_goals = self.pos_command_w.unsqueeze(1).expand(
                -1, self.cfg.trajectory_num_samples // env.num_envs, -1
            )
            self.all_spawn_locations = (
                env.scene.env_origins.clone()
                .unsqueeze(1)
                .expand(-1, self.cfg.trajectory_num_samples // env.num_envs, -1)
            )
            # perturb the goal positions
            self.all_goals = self.all_goals + torch.concat(
                [
                    sample_uniform(
                        self.cfg.goal_rand_x[0],
                        self.cfg.goal_rand_x[1],
                        (env.num_envs, self.cfg.trajectory_num_samples // env.num_envs, 1),
                        device=self.device,
                    ),
                    sample_uniform(
                        self.cfg.goal_rand_y[0],
                        self.cfg.goal_rand_y[1],
                        (env.num_envs, self.cfg.trajectory_num_samples // env.num_envs, 1),
                        device=self.device,
                    ),
                    torch.zeros((env.num_envs, self.cfg.trajectory_num_samples // env.num_envs, 1), device=self.device),
                ],
                dim=-1,
            )
            # perturb the start positions
            self.all_spawn_locations = self.all_spawn_locations + torch.concat(
                [
                    sample_uniform(
                        self.cfg.start_rand[0],
                        self.cfg.start_rand[1],
                        self.all_spawn_locations[..., :2].shape,
                        device=self.device,
                    ),
                    torch.ones(self.all_spawn_locations[..., 2].shape, device=self.device).unsqueeze(-1) * 0.1,
                ],
                dim=-1,
            )
            # Need to run the analysis to get the approx ideal distance from start to goal point
            if not self.cfg.project_onto_terrain:
                self._analysis = TerrainAnalysis(cfg=self.cfg.terrain_analysis, scene=self._env.scene)
            self._analysis.analyse()
            # get the ground truth path length
            self.all_path_length_command = self._analysis.shortest_path_lengths(
                self.all_spawn_locations.reshape(-1, 3), self.all_goals.reshape(-1, 3)
            ).reshape(self.all_goals.shape[:2])
            # check if want to project the goal points onto the terrain
            if self.cfg.project_onto_terrain:
                self.all_goals[:, :, 2] += self._analysis.get_height(self.all_goals[..., :2].reshape(-1, 2)).reshape(
                    self.all_goals.shape[:2]
                )
        self.not_updated_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_not_updated_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.nb_sampled_paths = 0
        self.path_idx_per_env = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "FixGoalCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base position in base frame. Shape is (num_envs, 3)."""
        return self.pos_command_b

    @property
    def analysis(self) -> TerrainAnalysis | TerrainAnalysisSingleton:
        """The terrain analysis object."""
        return self._analysis

    @property
    def all_path_completed(self) -> torch.Tensor | bool:
        """Check if all the sampled paths have been completed by a robot (i.e. all environments are no longer updated)"""
        return self.not_updated_envs.all() if self.cfg.trajectory_num_samples is not None else False

    @property
    def nb_generated_paths(self) -> int:
        """Number of paths that have been sampled from the environment"""
        return self.cfg.trajectory_num_samples if self.cfg.trajectory_num_samples is not None else 0

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Reset the command."""
        # reset only when all environments are reset
        if env_ids is None or len(env_ids) == self.num_envs:
            self.nb_sampled_paths = 0
            self.not_updated_envs.fill_(False)
            self.prev_not_updated_envs.fill_(False)

        return super().reset(env_ids=env_ids)

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new goal commands for the specified environments.

        Args:
            env_ids (Sequence[int]): The list of environment IDs to resample.
        """
        if self.cfg.trajectory_num_samples is not None:
            # save current state of not updated environments (necessary to log correct information for evaluation)
            self.prev_not_updated_envs = self.not_updated_envs.clone()
            # update non-updated environments
            self.not_updated_envs[env_ids[self.path_idx_per_env[env_ids] >= self.all_goals.shape[1]]] = True
            env_ids = env_ids[self.path_idx_per_env[env_ids] < self.all_goals.shape[1]]
            # Update goal and spawn locations
            self.pos_command_w[env_ids] = self.all_goals[env_ids, self.path_idx_per_env[env_ids]]
            self.pos_spawn_w[env_ids] = self.all_spawn_locations[env_ids, self.path_idx_per_env[env_ids]]
            # Update path length buffer
            self.path_length_command[env_ids] = self.all_path_length_command[env_ids, self.path_idx_per_env[env_ids]]
            # Update path index
            self.nb_sampled_paths += len(env_ids)
            self.path_idx_per_env[env_ids] += 1
        else:
            # perturb the start positions
            perturbation = torch.hstack([
                sample_uniform(self.cfg.start_rand[0], self.cfg.start_rand[1], (len(env_ids), 2), device=self.device),
                torch.zeros((len(env_ids), 1), device=self.device),
            ])
            perturbation[:, 2] = 0.1
            self.pos_spawn_w[env_ids] = self._env.scene.env_origins[env_ids] + perturbation

    def _update_command(self):
        """Re-target the position command to the current root position."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        target_vec[:, 2] = 0.0  # ignore z component
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)

    def _update_metrics(self):
        """Update metrics."""
        self.metrics["error_pos"] = torch.norm(self.pos_command_w - self.robot.data.root_pos_w[:, :3], dim=1)
