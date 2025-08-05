# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-module containing command generators for the position-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from scipy.spatial import KDTree
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from nav_suite.terrain_analysis import TerrainAnalysis, TerrainAnalysisSingleton

from .goal_command_base import GoalCommandBaseTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .consecutive_goal_command_cfg import ConsecutiveGoalCommandCfg


class ConsecutiveGoalCommand(GoalCommandBaseTerm):
    r"""Command that generates goal position commands based on terrain and defines the corresponding spawn locations.
    The goal commands are either sampled from RRT or from predefined fixed coordinates defined in the config.

    The goal coordinates/ commands are passed to the planners that generate the actual velocity commands.
    Goal coordinates are sampled in the world frame and then always transformed in the local robot frame.
    """

    cfg: ConsecutiveGoalCommandCfg
    """Configuration for the command."""

    def __init__(self, cfg: ConsecutiveGoalCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command class.

        Args:
            cfg: The configuration parameters for the command.
            env: The environment object.
        """
        super().__init__(cfg, env)

        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # -- goal commands in base frame: (x, y, z)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)

        self.path_length_command = torch.zeros(self.num_envs, device=self.device)

        # -- run terrain analysis (check if singleton is used and available)
        if (
            hasattr(self.cfg.terrain_analysis.class_type, "instance")
            and self.cfg.terrain_analysis.class_type.instance() is not None
        ):
            self._analysis = self.cfg.terrain_analysis.class_type.instance()
        else:
            self._analysis = self.cfg.terrain_analysis.class_type(self.cfg.terrain_analysis, scene=self._env.scene)
        self._analysis.analyse()
        # -- fit kd-tree on all the graph nodes to quickly find the closest node to the robot
        pruning_tensor = torch.ones(self._analysis.points.shape[0], dtype=bool, device=self.device)
        pruning_tensor[self._analysis.isolated_points_ids] = False
        self._kd_tree = KDTree(self._analysis.points[pruning_tensor].cpu().numpy())
        self._mapping_kd_tree_to_graph = torch.arange(pruning_tensor.sum(), device=self.device)
        self._mapping_kd_tree_to_graph += torch.cumsum(~pruning_tensor, 0)[pruning_tensor]

        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "ConsecutiveGoalCommandGenerator:\n"
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

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new goal commands for the specified environments.

        Args:
            env_ids (Sequence[int]): The list of environment IDs to resample.
        """
        # get the robot position for the environment
        robot_pos = self.robot.data.root_pos_w[env_ids, :3]

        # queery the closest node in the kd tree
        _, node_ids = self._kd_tree.query(robot_pos.cpu().numpy(), k=1)

        # map node ids to the graph
        node_ids = self._mapping_kd_tree_to_graph[node_ids]

        # sample a goal from the samples generated from the graph
        for env_id, node_id in zip(env_ids, node_ids):
            # get all samples
            samples = self._analysis.samples[self._analysis.samples[:, 0] == node_id]
            sample = samples[torch.randint(0, samples.shape[0], (1,))[0]].to(self.device)
            # sample a goal
            self.pos_command_w[env_id] = self._analysis.points[sample[1].long(), :3]
            # Update path length buffer
            self.path_length_command[env_id] = sample[2]

    def _update_command(self):
        """Re-target the position command to the current root position and heading."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        target_vec[:, 2] = 0.0  # ignore z component

        # update commands which are close to the goal
        goal_dist = torch.norm(target_vec, dim=1)
        close_goal = goal_dist < self.cfg.resample_distance_threshold
        if torch.any(close_goal):
            self._resample_command(torch.where(close_goal)[0])
            target_vec[close_goal] = self.pos_command_w[close_goal] - self.robot.data.root_pos_w[close_goal, :3]
            target_vec[close_goal, 2] = 0.0  # ignore z component

        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)

    def _update_metrics(self):
        """Update metrics."""
        self.metrics["error_pos"] = torch.norm(self.pos_command_w - self.robot.data.root_pos_w[:, :3], dim=1)
