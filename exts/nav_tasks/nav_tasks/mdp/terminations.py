# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import wrap_to_pi

from .commands import GoalCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def at_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.5,
    angle_threshold: float | None = None,
    speed_threshold: float | None = None,
    command_generator_term_name: str = "goal_command",
) -> torch.Tensor:
    """Terminate the episode when the goal is reached.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        distance_threshold: The distance threshold to the goal.
        speed_threshold: The speed threshold at the goal.

    Returns:
        Boolean tensor indicating whether the goal is reached.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_generator: GoalCommand = env.command_manager.get_term(command_generator_term_name)

    # Check conditions for termination
    distance_goal = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_generator.pos_command_w[:, :2], dim=1, p=2)
    within_distance = distance_goal < distance_threshold
    within_speed = True
    if speed_threshold is not None:
        abs_velocity = torch.norm(asset.data.root_vel_w[:, 0:6], dim=1, p=2)
        within_speed = abs_velocity < speed_threshold
    within_angle = True
    if angle_threshold is not None:
        if not hasattr(goal_cmd_generator, "heading_command_w"):
            omni.log.warn("The goal command generator does not have a heading_command_w attribute.")
        else:
            angle_goal = torch.abs(wrap_to_pi(asset.data.heading_w - goal_cmd_generator.heading_command_w))
            within_angle = angle_goal < angle_threshold
    return within_distance & within_speed & within_angle


def proportional_time_out(env: ManagerBasedRLEnv, max_speed: float = 1.0, safety_factor: float = 2.0) -> torch.Tensor:
    """Terminate the episode when it exceeds the episode length proportional to the goal distance for that robot.
    Min time of 20s"""
    # find the distance to the goal for each robot
    path_lengths = env.command_manager._terms["goal_command"].path_length_command  # type: ignore
    max_times = torch.max(torch.tensor(200.0), (safety_factor * path_lengths / max_speed) / env.step_dt)
    return env.episode_length_buf >= max_times


class StayedAtGoal(ManagerTermBase):
    """Terminate the episode when the goal is reached and maintained for enough time.

    The parameters are as follows:
        - time_threshold: The time to stay at the goal in seconds.
        - distance_threshold: The distance threshold to the goal.
        - angle_threshold: The angle threshold to the goal.
        - speed_threshold: The speed threshold at the goal.

    The last three are passed to the :func:`at_goal` function.

    Returns:
        Boolean tensor indicating whether the goal is reached and maintained for enough time.
    """

    # in ManagerBase._resolve_common_term_cfg any func of that is a callable class inheriting from ManagerTermBase are initialised with the signature (cfg, env)
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # create the buffer to store the time at goal for each environment
        self.time_at_goal = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        time_threshold: float,
        distance_threshold: float,
        angle_threshold: float,
        speed_threshold: float,
    ) -> torch.Tensor:
        # check if the goal is reached at this step
        currently_at_goal = at_goal(env, distance_threshold=distance_threshold, angle_threshold=angle_threshold, speed_threshold=speed_threshold)  # type: ignore

        # update the time at goal
        self.time_at_goal[currently_at_goal] += env.step_dt
        self.time_at_goal[~currently_at_goal] = 0.0

        # check if the time at goal exceeds the threshold, don't worry about resetting buffer this is called later
        return self.time_at_goal > time_threshold

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the manager term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        self.time_at_goal[env_ids] = 0.0
