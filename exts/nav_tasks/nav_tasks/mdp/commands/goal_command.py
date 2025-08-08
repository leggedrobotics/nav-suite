# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-module containing command generators for the position-based locomotion task."""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING

import omni.log
import omni.usd
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, wrap_to_pi, yaw_quat

from nav_suite.collectors import TrajectorySampling
from nav_suite.terrain_analysis import TerrainAnalysis, TerrainAnalysisSingleton

from .goal_command_base import CYLINDER_MARKER_CFG, GoalCommandBaseTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .goal_command_cfg import GoalCommandCfg


class GoalCommand(GoalCommandBaseTerm):
    r"""Command that generates goal position commands based on terrain and defines the corresponding spawn locations.
    The goal commands are either sampled from RRT or from predefined fixed coordinates defined in the config.

    The goal coordinates/ commands are passed to the planners that generate the actual velocity commands.
    Goal coordinates are sampled in the world frame and then always transformed in the local robot frame.
    """

    cfg: GoalCommandCfg
    """Configuration for the command."""

    def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command class.

        Args:
            cfg: The configuration parameters for the command.
            env: The environment object.

        Raises:
            AssertionError: If both autonomous resampling and infinite sampling are enabled.
        """
        super().__init__(cfg, env)

        # -- goal commands in base frame: (x, y, z)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)

        # -- heading command
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

        # -- path length of the start-goal pairs
        self.path_length_command = torch.zeros(self.num_envs, device=self.device)

        # -- spawn locations (x, y, z, heading)
        self.pos_spawn_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_spawn_w = torch.zeros(self.num_envs, device=self.device)

        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

        # -- evaluation - monitor not updated environments
        self._not_updated_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_not_updated_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.nb_sampled_paths = 0

        # -- tracking when the trajectory config was last updated, for use in curriculum updates.
        self.last_update_config_env_step = 0

        # init trajectory sampling object
        if self.cfg.subterrain_sampling and self._env.scene.terrain.cfg.terrain_type == "usd":
            # execute a terrain analysis for each subterrain
            assert self.cfg.subterrain_analysis_cfgs is not None, "Subterrain analysis configs are not defined."
            self._traj_sampling = {}
            for subterrain_prim_path, subterrain_cfg in self.cfg.subterrain_analysis_cfgs.items():
                # transform bounding box based on the subterrain prim path (if given)
                if "terrain_bounding_box" in subterrain_cfg:
                    path_prim = self._env.scene.stage.GetPrimAtPath(subterrain_prim_path)
                    # NOTE: remove scale, assume scale is already applied to the bounding box
                    transform = np.asarray(omni.usd.get_world_transform_matrix(path_prim).RemoveScaleShear()).T
                    # corner points of the bounding box according to the bbox point definition [x_min, y_min, x_max, y_max]
                    box_points = np.array([
                        [
                            subterrain_cfg["terrain_bounding_box"][0],
                            subterrain_cfg["terrain_bounding_box"][1],
                            0.0,
                            1.0,
                        ],
                        [
                            subterrain_cfg["terrain_bounding_box"][2],
                            subterrain_cfg["terrain_bounding_box"][1],
                            0.0,
                            1.0,
                        ],
                        [
                            subterrain_cfg["terrain_bounding_box"][0],
                            subterrain_cfg["terrain_bounding_box"][3],
                            0.0,
                            1.0,
                        ],
                        [
                            subterrain_cfg["terrain_bounding_box"][2],
                            subterrain_cfg["terrain_bounding_box"][3],
                            0.0,
                            1.0,
                        ],
                    ])
                    box_points = np.dot(transform, box_points.T).T
                    subterrain_cfg["terrain_bounding_box"] = [
                        np.min(box_points[:, 0]),
                        np.min(box_points[:, 1]),
                        np.max(box_points[:, 0]),
                        np.max(box_points[:, 1]),
                    ]
                # set the terrain analysis cfg for the subterrain
                sampling_cfg = deepcopy(self.cfg.traj_sampling)
                sampling_cfg.terrain_analysis = sampling_cfg.terrain_analysis.replace(**subterrain_cfg)
                self._traj_sampling[subterrain_prim_path] = TrajectorySampling(cfg=sampling_cfg, scene=self._env.scene)
        else:
            self._traj_sampling = {"/World": TrajectorySampling(cfg=self.cfg.traj_sampling, scene=self._env.scene)}

        if self.cfg.terrain_level_sampling:
            assert (
                self._env.scene.terrain.cfg.terrain_type == "generator"
            ), "Terrain level sampling is only supported for generator terrains."

        # -- run terrain analysis and sample first trajectories
        self.sample_trajectories()

        print(self.__str__())

    def __str__(self) -> str:
        msg = "GoalCommandGenerator:\n"
        msg += f"\tCommand dimension:\t {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range:\t {self.cfg.resampling_time_range}\n"
        msg += f"\tSampling mode:\t\t {self.cfg.sampling_mode}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base pose in base frame. Shape is (num_envs, 4)."""
        return torch.cat((self.pos_command_b, self.heading_command_b.unsqueeze(-1)), dim=1)

    @property
    def path_sampled_ratio(self) -> float:
        """Percentage of the sampled paths that have already been sampled and assigned to a robot"""
        return self.nb_sampled_paths / self._nb_generated_paths

    @property
    def all_path_completed(self) -> torch.Tensor:
        """Check if all the sampled paths have been completed by a robot (i.e. all environments are no longer updated)"""
        return self._not_updated_envs.all()

    @property
    def nb_generated_paths(self) -> int:
        """Number of paths that have been sampled from the environment"""
        return self._nb_generated_paths

    @property
    def analysis(self, prim_path: str = "/World") -> TerrainAnalysis | TerrainAnalysisSingleton:
        """The terrain analysis object."""
        return self._traj_sampling[prim_path].terrain_analyser

    """
    Operations
    """

    def update_trajectory_config(
        self, num_pairs: int | None = None, min_path_length: float | None = None, max_path_length: float | None = None
    ):
        """Update the trajectory configuration for sampling start-goal pairs.

        Args:
            num_pairs (int, optional): Number of start-goal pairs to sample. Defaults to 10.
            min_path_length (float, optional): Minimum path length between start and goal. Defaults to 0.
            max_path_length (float, optional): Maximum path length between start and goal. Defaults to np.inf.
        """
        # Update trajectory config
        if num_pairs is not None:
            self.cfg.num_pairs = num_pairs
        if min_path_length is not None:
            self.cfg.path_length_range[0] = min_path_length
        if max_path_length is not None:
            self.cfg.path_length_range[1] = max_path_length
        self.last_update_config_env_step = self._env.common_step_counter
        self.sample_trajectories()

    def sample_trajectories(self):
        """Sample trajectories"""
        # Sample new start-goal pairs from RRT
        omni.log.info(
            "Sampling start-goal pairs with following configuration:\n"
            f"\tNumber of start-goal pairs: {self.cfg.num_pairs}\n"
            f"\tMinimum path length: {self.cfg.path_length_range[0]}\n"
            f"\tMaximum path length: {self.cfg.path_length_range[1]}"
        )

        if self.cfg.subterrain_sampling and self._env.scene.terrain.cfg.terrain_type == "generator":
            # Check if the terrain being used has terrain_origins defined. If not, we can't use
            # the sample_paths_by_terrain function.
            assert (
                self._traj_sampling["/World"].scene.terrain.terrain_origins is not None
            ), "Subterrain origins are not defined for the terrain being used. This typically happens for USD terrains."
            # paths have the entries:
            # [start_x, start_y, start_z, goal_x, goal_y, goal_z, path_length] with shape
            # (num_terrain_levels, num_terrain_types, num_pairs, 7)
            # Note that this module assumes start_z and goal_z are at the robot's base height above the terrain.
            self.paths = self._traj_sampling["/World"].sample_paths_by_terrain(
                num_paths=self.cfg.num_pairs,
                min_path_length=self.cfg.path_length_range[0],
                max_path_length=self.cfg.path_length_range[1],
                terrain_level_sampling=self.cfg.terrain_level_sampling,
            )

            if self.cfg.terrain_level_sampling:
                # reshape paths to (row, num_paths, 7)
                self.paths = self.paths.reshape(self.paths.shape[0], -1, 7)
                if self.paths.shape[1] < self.num_envs:
                    omni.log.warn("Number of paths sampled is less than the number of environments. Will repeat paths.")
                    self.paths = self.paths.repeat(1, (self.num_envs // self.paths.shape[1]) + 1, 1)
            else:
                self.paths = self.paths.reshape(-1, 7)
        elif self.cfg.subterrain_sampling and self._env.scene.terrain.cfg.terrain_type == "usd":
            # execute a terrain analysis for each subterrain
            paths = []
            num_paths_sampled = 0
            for subterrain_prim_path, subterrain_traj_sampler in self._traj_sampling.items():
                # set the terrain analysis cfg for the subterrain
                omni.log.info(f"Sampling paths for subterrain: {subterrain_prim_path}")
                paths.append(
                    subterrain_traj_sampler.sample_paths(
                        num_paths=self.cfg.num_pairs,
                        min_path_length=self.cfg.path_length_range[0],
                        max_path_length=self.cfg.path_length_range[1],
                    )
                )
                num_paths_sampled += len(paths[-1])

            # stack paths interleaved (i.e. subterrain 1, subterrain 2, subterrain 1, subterrain 2, ...)
            self.paths = torch.stack(paths, dim=1).reshape(-1, paths[0].shape[1])
            subterrain_idx = [torch.ones(curr_path.shape[0], 1) * idx for idx, curr_path in enumerate(paths)]
            self.subterrain_idx = torch.stack(subterrain_idx, dim=1).reshape(-1)
        else:
            # paths have the entries:
            # [start_x, start_y, start_z, goal_x, goal_y, goal_z, path_length] with shape (num_paths, 7)
            self.paths = self._traj_sampling["/World"].sample_paths(
                num_paths=self.cfg.num_pairs,
                min_path_length=self.cfg.path_length_range[0],
                max_path_length=self.cfg.path_length_range[1],
            )

        # Update number of paths
        if self.cfg.terrain_level_sampling:
            self._nb_generated_paths = self.paths.reshape(-1, 7).shape[0]
            self.nb_sampled_paths_per_row = torch.zeros(self.paths.shape[0], device="cpu", dtype=torch.long)
        else:
            self._nb_generated_paths = self.paths.shape[0]
            self.nb_sampled_paths_per_row = None
        self.nb_sampled_paths = 0

        omni.log.info(": Sampling has finished.")

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Reset the command."""
        # reset only when all environments are reset
        if env_ids is None or len(env_ids) == self.num_envs:
            self.nb_sampled_paths = 0
            self._not_updated_envs.fill_(False)
            self._prev_not_updated_envs.fill_(False)

        return super().reset(env_ids=env_ids)

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new goal commands for the specified environments.

        Args:
            env_ids (Sequence[int]): The list of environment IDs to resample.
        """

        if self.cfg.sampling_mode == "bounded":
            # save current state of not updated environments (necessary to log correct information for evaluation)
            self._prev_not_updated_envs = self._not_updated_envs.clone()

            # if no infinite sampling, only update for as many environment as there are new trajectories
            if self.cfg.terrain_level_sampling:
                # get the paths for the corresponding terrain level
                curr_levels, level_counts = torch.unique(
                    self._env.scene.terrain.terrain_levels[env_ids], return_counts=True
                )

                sample = torch.zeros(len(env_ids), 7, device=self.paths.device)
                for level, count in zip(curr_levels.tolist(), level_counts.tolist()):
                    level_env_ids = (self._env.scene.terrain.terrain_levels[env_ids] == level).to(self.paths.device)
                    if count > (self.paths.shape[1] - self.nb_sampled_paths_per_row[level]):
                        # update non-updated environments
                        self._not_updated_envs[
                            env_ids[
                                level_env_ids[
                                    self.nb_sampled_paths_per_row[level] : self.nb_sampled_paths_per_row[level] + count
                                ]
                            ]
                        ] = True
                        # only update for as many environments as there are new trajectories
                        level_env_ids[: max(self.paths.shape[1] - self.nb_sampled_paths_per_row[level], 0)] = False

                    sample[level_env_ids] = self.paths[
                        level, self.nb_sampled_paths_per_row[level] : self.nb_sampled_paths_per_row[level] + count
                    ]
            else:
                if len(env_ids) > self._nb_generated_paths - self.nb_sampled_paths:
                    # update non-updated environments
                    self._not_updated_envs[env_ids[self._nb_generated_paths - self.nb_sampled_paths :]] = True
                    # only update for as many environments as there are new trajectories
                    env_ids = env_ids[: max(self._nb_generated_paths - self.nb_sampled_paths, 0)]
                sample = self.paths[self.nb_sampled_paths : self.nb_sampled_paths + len(env_ids)]
            self.nb_sampled_paths += len(env_ids)

        elif self.cfg.sampling_mode == "autonomous":
            if self.cfg.terrain_level_sampling:
                # get the paths for the corresponding terrain level
                curr_levels, level_counts = torch.unique(
                    self._env.scene.terrain.terrain_levels[env_ids], return_counts=True
                )

                sample = torch.zeros(len(env_ids), 7, device=self.paths.device)
                if any([
                    count > (self.paths.shape[1] - self.nb_sampled_paths_per_row[level])
                    for level, count in zip(curr_levels, level_counts)
                ]):
                    self.sample_trajectories()
                    self.nb_sampled_paths = 0

                for level, count in zip(curr_levels.tolist(), level_counts.tolist()):
                    sample[self._env.scene.terrain.terrain_levels[env_ids] == level] = self.paths[
                        level, self.nb_sampled_paths_per_row[level] : self.nb_sampled_paths_per_row[level] + count
                    ]
                    self.nb_sampled_paths_per_row[level] += count
            else:
                if len(env_ids) > self._nb_generated_paths - self.nb_sampled_paths:
                    self.sample_trajectories()
                    self.nb_sampled_paths = 0

                sample = self.paths[self.nb_sampled_paths : self.nb_sampled_paths + len(env_ids)]

            self.nb_sampled_paths += len(env_ids)

        elif self.cfg.sampling_mode == "infinite":
            if self.cfg.terrain_level_sampling:
                curr_levels, level_counts = torch.unique(
                    self._env.scene.terrain.terrain_levels[env_ids], return_counts=True
                )
                sample = torch.zeros(len(env_ids), 7, device=self.paths.device)
                for level, count in zip(curr_levels, level_counts):
                    sample_idx = torch.randperm(self._nb_generated_paths)[:count]
                    sample[self._env.scene.terrain.terrain_levels[env_ids] == level] = self.paths[level, sample_idx]
            else:
                sample_idx = torch.randperm(self._nb_generated_paths)[: len(env_ids)]
                sample = self.paths[sample_idx]

        else:
            raise ValueError(f"Invalid sampling mode: {self.cfg.sampling_mode}")

        # Update command buffers
        self.pos_command_w[env_ids] = sample[:, 3:6].to(self._env.device)

        # Update spawn locations and heading buffer
        self.pos_spawn_w[env_ids] = sample[:, :3].to(self._env.device)
        self.pos_spawn_w[env_ids, 2] += self.cfg.z_offset_spawn

        # Calculate the spawn heading based on the goal position
        self.heading_spawn_w[env_ids] = torch.atan2(
            self.pos_command_w[env_ids, 1] - self.pos_spawn_w[env_ids, 1],
            self.pos_command_w[env_ids, 0] - self.pos_spawn_w[env_ids, 0],
        )
        # Calculate the goal heading based on the goal position
        self.heading_command_w[env_ids] = torch.atan2(
            self.pos_command_w[env_ids, 1] - self.pos_spawn_w[env_ids, 1],
            self.pos_command_w[env_ids, 0] - self.pos_spawn_w[env_ids, 0],
        )

        # Update path length buffer
        self.path_length_command[env_ids] = sample[:, 6].to(self._env.device)

        # NOTE: the reset event is called before the new goal commands are generated, i.e. the spawn locations are
        # updated before the new goal commands are generated. To repsawn with the correct locations, we call here the
        # update spawn locations function
        if self.cfg.reset_pos_term_name:
            reset_term_idx = self._env.event_manager.active_terms["reset"].index(self.cfg.reset_pos_term_name)
            self._env.event_manager._mode_term_cfgs["reset"][reset_term_idx].func(
                self._env, env_ids, **self._env.event_manager._mode_term_cfgs["reset"][reset_term_idx].params
            )

    def _update_command(self):
        """Re-target the position command to the current root position and heading."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        target_vec[:, 2] = 0.0  # ignore z component
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)

        # update the heading command in the base frame
        # heading_w is angle world x axis to robot base x axis
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

    def _update_metrics(self):
        """Update metrics."""
        self.metrics["error_pos"] = torch.norm(self.pos_command_w - self.robot.data.root_pos_w[:, :3], dim=1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set the debug visualization for the command.

        Args:
            debug_vis (bool): Whether to enable debug visualization.
        """
        # init all debug markers common for all goal command generators
        super()._set_debug_vis_impl(debug_vis)

        # create markers if necessary for the first time
        # for each marker type check that the correct command properties exist eg. need spawn position for spawn marker
        if debug_vis:
            if not hasattr(self, "box_spawn_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/position_goal"
                marker_cfg.markers["cuboid"].size = (0.3, 0.3, 0.3)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                self.box_spawn_visualizer = VisualizationMarkers(marker_cfg)
                self.box_spawn_visualizer.set_visibility(True)
            if not hasattr(self, "goal_heading_visualizer"):
                marker_cfg = CYLINDER_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_heading"
                marker_cfg.markers["cylinder"].height = 1
                marker_cfg.markers["cylinder"].radius = 0.03
                marker_cfg.markers["cylinder"].visual_material.diffuse_color = (0, 0, 1.0)
                self.goal_heading_visualizer = VisualizationMarkers(marker_cfg)
                self.goal_heading_visualizer.set_visibility(True)
        else:
            if hasattr(self, "box_spawn_visualizer"):
                self.box_spawn_visualizer.set_visibility(False)
            if hasattr(self, "heading_goal_visualizer"):
                self.goal_heading_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event, env_ids: Sequence[int] | None = None):
        """Callback function for the debug visualization."""
        if env_ids is None:
            env_ids = slice(None)

        # call the base class debug visualization
        super()._debug_vis_callback(event, env_ids)

        # update spawn marker if it exists
        self.box_spawn_visualizer.visualize(self.pos_spawn_w[env_ids])

        # command heading marker
        orientations = quat_from_euler_xyz(
            torch.zeros_like(self.heading_command_w),
            torch.zeros_like(self.heading_command_w),
            self.heading_command_w,
        )
        translations = self.pos_command_w + quat_apply(
            orientations, torch.Tensor([0.5, 0, 0]).to(self.device).repeat(orientations.shape[0], 1)
        )
        self.goal_heading_visualizer.visualize(translations[env_ids], orientations[env_ids])
